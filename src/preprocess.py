import os
import hashlib
from datasets import load_dataset, concatenate_datasets

def format_prompt(prompt):
    """Formats a raw prompt into the Llama-2 chat template."""
    return f"<s>[INST] {prompt} [/INST] "

def parse_hh_rlhf(example):
    """Parses a single example from the Anthropic/hh-rlhf dataset."""
    try:
        chosen_text = example['chosen'].strip()
        rejected_text = example['rejected'].strip()

        # Find the last assistant turn in the 'chosen' text to define the prompt
        last_assistant_pos = chosen_text.rfind('Assistant:')
        if last_assistant_pos == -1:
            return None
        
        raw_prompt = chosen_text[:last_assistant_pos].strip().replace('Human:', '').strip()
        chosen_answer = chosen_text[last_assistant_pos:].replace('Assistant:', '').strip()
        
        # The rejected answer is just the last turn
        rejected_answer = rejected_text.split('Assistant:')[-1].strip()

        return {'prompt': raw_prompt, 'chosen': chosen_answer, 'rejected': rejected_answer}
    except (TypeError, AttributeError, IndexError):
        return None

def parse_argilla(example):
    """Parses a single example from the argilla/dpo-mix-7k dataset."""
    try:
        prompt = next(turn['content'] for turn in example['chosen'] if turn['role'] == 'user')
        chosen = next(turn['content'] for turn in example['chosen'] if turn['role'] == 'assistant')
        rejected = next(turn['content'] for turn in example['rejected'] if turn['role'] == 'assistant')
        return {'prompt': prompt, 'chosen': chosen, 'rejected': rejected}
    except (StopIteration, TypeError, KeyError):
        return None

def load_and_prepare_datasets(config, tokenizer):
    """Loads, preprocesses, and splits the datasets based on the config."""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Warning: HF_TOKEN environment variable not set. May fail to download gated models/datasets.")

    processed_datasets = []
    for d_config in config['datasets']:
        try:
            split = d_config.get("split", "train")
            dataset = load_dataset(d_config['name'], split=split, token=hf_token)
            
            if d_config['name'] == 'Anthropic/hh-rlhf':
                dataset = dataset.map(parse_hh_rlhf, num_proc=4)
            elif d_config['name'] == 'argilla/dpo-mix-7k':
                dataset = dataset.map(parse_argilla, num_proc=4)
            
            dataset = dataset.filter(lambda x: x is not None and all(x.values()))
            processed_datasets.append(dataset)
        except Exception as e:
            print(f"Failed to load or process dataset {d_config['name']}: {e}")
            continue

    if not processed_datasets:
        raise ValueError("No datasets could be loaded.")

    combined_dataset = concatenate_datasets(processed_datasets).shuffle(seed=config.get('seed', 42))

    def tokenize_and_format(examples):
        prompts = [format_prompt(p) for p in examples['prompt']]
        chosen_responses = [c + tokenizer.eos_token for c in examples['chosen']]
        rejected_responses = [r + tokenizer.eos_token for r in examples['rejected']]
        
        tokenized_prompts = tokenizer(prompts, add_special_tokens=False)
        tokenized_chosen = tokenizer(chosen_responses, add_special_tokens=False)
        tokenized_rejected = tokenizer(rejected_responses, add_special_tokens=False)

        batch = {
            'prompt_input_ids': [], 'chosen_input_ids': [], 'rejected_input_ids': [],
            'chosen_attention_mask': [], 'rejected_attention_mask': []
        }
        max_length = config['data_params']['max_seq_length']

        for i in range(len(prompts)):
            prompt_ids = tokenized_prompts['input_ids'][i]
            chosen_ids = tokenized_chosen['input_ids'][i]
            rejected_ids = tokenized_rejected['input_ids'][i]
            
            chosen_len = len(prompt_ids) + len(chosen_ids)
            rejected_len = len(prompt_ids) + len(rejected_ids)

            if chosen_len > max_length: continue
            if rejected_len > max_length: continue

            batch['prompt_input_ids'].append(prompt_ids)
            batch['chosen_input_ids'].append(prompt_ids + chosen_ids)
            batch['rejected_input_ids'].append(prompt_ids + rejected_ids)
            batch['chosen_attention_mask'].append([1] * (len(prompt_ids) + len(chosen_ids)))
            batch['rejected_attention_mask'].append([1] * (len(prompt_ids) + len(rejected_ids)))

        return batch

    tokenized_dataset = combined_dataset.map(
        tokenize_and_format, batched=True, remove_columns=combined_dataset.column_names, num_proc=4
    )

    val_percentage = config['data_params']['val_split_percentage']
    def is_validation(example, percentage):
        prompt_str = tokenizer.decode(example['prompt_input_ids'][:50])
        hash_val = int(hashlib.sha1(prompt_str.encode("utf-8")).hexdigest(), 16)
        return hash_val % 100 < percentage

    val_dataset = tokenized_dataset.filter(lambda x: is_validation(x, val_percentage))
    train_dataset = tokenized_dataset.filter(lambda x: not is_validation(x, val_percentage))

    print(f"Dataset prepared: {len(train_dataset)} training samples, {len(val_dataset)} validation samples.")
    return train_dataset, val_dataset
