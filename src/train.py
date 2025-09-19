import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_scheduler
from peft import get_peft_model, LoraConfig
from bitsandbytes.optim import AdamW8bit as AdamW
from tqdm import tqdm
from collections import defaultdict

# --- Model Loading ---
def get_model_and_tokenizer(model_name, lora_config_dict, training_args):
    """Initializes the model and tokenizer with quantization and LoRA."""
    hf_token = os.getenv("HF_TOKEN")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            token=hf_token,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
    except Exception as e:
        raise RuntimeError(f"Failed to load model or tokenizer '{model_name}': {e}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(**lora_config_dict)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return model, tokenizer

# --- Loss Calculation ---
def _get_batch_logps(model, input_ids, attention_mask, prompt_lengths):
    """Computes the log probabilities of the sequences in a batch."""
    # Shift labels for autoregressive training
    labels = input_ids.clone()
    labels[labels == model.config.pad_token_id] = -100 # Ignore pad tokens in loss

    # Mask prompt tokens
    for i in range(len(prompt_lengths)):
        labels[i, :prompt_lengths[i]] = -100

    with torch.no_grad(): # Don't need gradients for logp calculation itself
      model.eval()
      outputs = model(input_ids, attention_mask=attention_mask)
      model.train()
    
    logits = outputs.logits
    # Get per-token log probabilities
    log_probs = F.log_softmax(logits, dim=-1)
    # Gather the logps of the true tokens
    labels_clamped = labels.clamp(min=0)
    token_logps = torch.gather(log_probs, -1, labels_clamped.unsqueeze(-1)).squeeze(-1)

    # Replace ignored indices (-100) with 0 for summation
    token_logps[labels == -100] = 0
    # Sum over sequence length to get sequence log probability
    seq_logps = token_logps.sum(dim=1)
    return seq_logps, token_logps, logits

def compute_loss(model, batch, config):
    """Computes the total loss for a batch based on the configured loss type."""
    loss_type = config['loss_type']
    params = config.get('loss_params', {})

    # Move batch to device
    chosen_input_ids = batch['chosen_input_ids'].to(model.device)
    chosen_attention_mask = batch['chosen_attention_mask'].to(model.device)
    rejected_input_ids = batch['rejected_input_ids'].to(model.device)
    rejected_attention_mask = batch['rejected_attention_mask'].to(model.device)
    prompt_lengths = [len(p) for p in batch['prompt_input_ids']]

    chosen_seq_logps, _, _ = _get_batch_logps(model, chosen_input_ids, chosen_attention_mask, prompt_lengths)
    rejected_seq_logps, _, _ = _get_batch_logps(model, rejected_input_ids, rejected_attention_mask, prompt_lengths)

    # DPO Loss (serves as L_pref for GAPO and baseline)
    beta = params.get('beta', 0.1)
    log_odds = (chosen_seq_logps - rejected_seq_logps) * beta
    l_pref = -F.logsigmoid(log_odds).mean()

    if loss_type == 'dpo':
        return l_pref, {'dpo_loss': l_pref.item()}

    elif loss_type == 'gapo':
        # --- GAPO Specific Loss Calculation ---
        lambda_tok = params.get('lambda_tok', 0.1)
        kappa = params.get('kappa', 1.0)

        # Re-run forward pass with gradients for the chosen response to get token probabilities
        model.train()
        outputs_chosen = model(chosen_input_ids, attention_mask=chosen_attention_mask)
        logits_chosen = outputs_chosen.logits
        probs_chosen = F.softmax(logits_chosen, dim=-1)

        # Get logps and probabilities for chosen and rejected answers
        with torch.no_grad():
            model.eval()
            outputs_rejected = model(rejected_input_ids, attention_mask=rejected_attention_mask)
            log_probs_rejected = F.log_softmax(outputs_rejected.logits, dim=-1)
            model.train()
        
        log_probs_chosen = F.log_softmax(logits_chosen, dim=-1)

        labels_chosen = chosen_input_ids.clone()
        labels_rejected = rejected_input_ids.clone()
        labels_chosen[labels_chosen == model.config.pad_token_id] = -100
        labels_rejected[labels_rejected == model.config.pad_token_id] = -100

        # Align sequences by padding to the max length
        max_len = max(chosen_input_ids.shape[1], rejected_input_ids.shape[1])
        
        def pad_and_gather(log_probs, labels, prompt_len, target_len):
            if log_probs.dim() == 2:
                log_probs = log_probs.unsqueeze(0)
            if labels.dim() == 1:
                labels = labels.unsqueeze(0)
            
            pad_len = target_len - log_probs.shape[1]
            if pad_len > 0:
                padded_log_probs = F.pad(log_probs, (0, 0, 0, pad_len))
                padded_labels = F.pad(labels, (0, pad_len), value=-100)
            else:
                padded_log_probs = log_probs[:, :target_len]
                padded_labels = labels[:, :target_len]
            # Mask prompt
            padded_labels[:, :prompt_len] = -100
            labels_clamped = padded_labels.clamp(min=0)
            gathered = torch.gather(padded_log_probs, -1, labels_clamped.unsqueeze(-1)).squeeze(-1)
            gathered[padded_labels == -100] = 0 # Set ignored tokens to 0
            return gathered.squeeze(0), (padded_labels != -100).squeeze(0)

        # This part is complex. We compute per-token gains.
        gains = []
        p_chosen_list = []
        answer_masks = []
        for i in range(len(prompt_lengths)):
            prompt_len = prompt_lengths[i]
            logp_c, mask_c = pad_and_gather(log_probs_chosen[i], labels_chosen[i], prompt_len, max_len)
            logp_r, _ = pad_and_gather(log_probs_rejected[i], labels_chosen[i], prompt_len, max_len) # use chosen labels for rejected too
            p_c, _ = pad_and_gather(probs_chosen[i], labels_chosen[i], prompt_len, max_len)
            
            gains.append(logp_c - logp_r)
            p_chosen_list.append(p_c)
            answer_masks.append(mask_c)

        g_t = torch.stack(gains)
        p_chosen_t = torch.stack(p_chosen_list)
        answer_mask = torch.stack(answer_masks)
        
        with torch.no_grad():
            weights = F.softmax(kappa * g_t, dim=1)
            weights = weights * answer_mask # Ensure weights are zero for padding
            
            # Normalize weights to sum to 1 over non-padded tokens
            weights_sum = weights.sum(dim=1, keepdim=True)
            weights = weights / (weights_sum + 1e-8)
            
            w_bar = (1.0 / (answer_mask.sum(dim=1, keepdim=True) + 1e-8)) - weights

        # Brier loss component
        l_tok_positive = torch.sum(weights * (1 - p_chosen_t)**2, dim=1)
        l_tok_negative = torch.sum(w_bar * p_chosen_t**2, dim=1)
        l_tok = (l_tok_positive + l_tok_negative).mean()

        total_loss = l_pref + lambda_tok * l_tok
        loss_dict = {
            'total_loss': total_loss.item(),
            'l_pref': l_pref.item(),
            'l_tok': l_tok.item()
        }
        return total_loss, loss_dict

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

# --- Training Loop ---
def collate_fn(batch):
    """Pad sequences to the max length in a batch."""
    collated = defaultdict(list)
    max_len_chosen = max(len(item['chosen_input_ids']) for item in batch)
    max_len_rejected = max(len(item['rejected_input_ids']) for item in batch)

    # Use a dummy pad_token_id, assuming tokenizer is available in main scope
    pad_token_id = 0 # Should be set from tokenizer

    for item in batch:
        for key, value in item.items():
            if 'input_ids' in key:
                max_len = max_len_chosen if 'chosen' in key else max_len_rejected
                padded = value + [pad_token_id] * (max_len - len(value))
                collated[key].append(padded)
            elif 'attention_mask' in key:
                max_len = max_len_chosen if 'chosen' in key else max_len_rejected
                padded = value + [0] * (max_len - len(value))
                collated[key].append(padded)
            else: # prompt_input_ids
                 collated[key].append(value)
    
    return {key: torch.tensor(val) for key, val in collated.items() if 'prompt' not in key} | {'prompt_input_ids': collated['prompt_input_ids']}

def train_model(config, model, tokenizer, train_dataset, val_dataset):
    """Main training loop for the model."""
    training_args = config['training_args']
    
    # Use a local collate function with the correct pad_token_id
    def local_collate_fn(batch):
        collated = defaultdict(list)
        max_len_chosen = max(len(item['chosen_input_ids']) for item in batch)
        max_len_rejected = max(len(item['rejected_input_ids']) for item in batch)

        for item in batch:
            for key, value in item.items():
                if 'input_ids' in key:
                    max_len = max_len_chosen if 'chosen' in key else max_len_rejected
                    padded = value + [tokenizer.pad_token_id] * (max_len - len(value))
                    collated[key].append(padded)
                elif 'attention_mask' in key:
                    max_len = max_len_chosen if 'chosen' in key else max_len_rejected
                    padded = value + [0] * (max_len - len(value))
                    collated[key].append(padded)
                else: # prompt_input_ids
                    collated[key].append(value)
        
        return {key: torch.tensor(val) for key, val in collated.items() if 'prompt' not in key} | {'prompt_input_ids': collated['prompt_input_ids']}


    train_loader = DataLoader(
        train_dataset, 
        batch_size=training_args['per_device_train_batch_size'], 
        shuffle=True, 
        collate_fn=local_collate_fn
    )

    optimizer = AdamW(model.parameters(), lr=training_args['learning_rate'], betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
    
    num_training_steps = training_args.get('max_steps', len(train_loader) * training_args.get('num_train_epochs', 1) // training_args['gradient_accumulation_steps'])

    scheduler = get_scheduler(
        name=training_args['lr_scheduler_type'],
        optimizer=optimizer,
        num_warmup_steps=training_args['warmup_steps'],
        num_training_steps=num_training_steps
    )

    progress_bar = tqdm(range(num_training_steps))
    global_step = 0

    model.train()
    for epoch in range(training_args.get('num_train_epochs', 100)):
        for i, batch in enumerate(train_loader):
            loss, loss_dict = compute_loss(model, batch, training_args)
            loss = loss / training_args['gradient_accumulation_steps']
            loss.backward()

            if (i + 1) % training_args['gradient_accumulation_steps'] == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                global_step += 1

                if global_step % training_args['logging_steps'] == 0:
                    progress_bar.set_postfix(loss_dict)

            if global_step >= num_training_steps:
                break
        if global_step >= num_training_steps:
            break

    print("Training finished.")
    # Save final model
    final_checkpoint_dir = os.path.join(training_args['output_dir'], 'final_checkpoint')
    model.save_pretrained(final_checkpoint_dir)
    tokenizer.save_pretrained(final_checkpoint_dir)
    print(f"Final model saved to {final_checkpoint_dir}")

    return model
