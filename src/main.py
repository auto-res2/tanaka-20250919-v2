import argparse
import yaml
import os
import torch
import json
import time

from .preprocess import load_and_prepare_datasets
from .train import get_model_and_tokenizer, train_model
from .evaluate import run_evaluation

def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic algorithms are used
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)

def load_config(config_path):
    """Load YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        exit(1)

def run_experiment(config, is_smoke_test=False):
    """Orchestrates a single or a series of experimental runs."""
    # Create output directories
    base_output_dir = config['base_training_args']['output_dir'] if not is_smoke_test else config['training_args']['output_dir']
    os.makedirs(base_output_dir, exist_ok=True)
    os.makedirs(os.path.join(base_output_dir, 'images'), exist_ok=True)

    if is_smoke_test:
        print("--- Running Smoke Test ---")
        set_seed(config['training_args']['seed'])
        run_single_trial(config)
    else:
        print("--- Running Full Experiment ---")
        all_results = []
        for i, run_config in enumerate(config['experiment_runs']):
            print(f"\n>>> Starting Run {i+1}/{len(config['experiment_runs'])}: {run_config['name']} <<<")
            
            # Merge base config with run-specific config
            trial_config = config.copy()
            trial_config['training_args'] = config['base_training_args'].copy()
            trial_config['training_args']['loss_type'] = run_config['loss_type']
            trial_config['training_args']['loss_params'] = run_config['loss_params']
            trial_config['training_args']['seed'] = run_config['seed']
            trial_config['training_args']['output_dir'] = os.path.join(config['base_training_args']['output_dir'], run_config['name'])

            set_seed(run_config['seed'])
            results = run_single_trial(trial_config)
            results['run_name'] = run_config['name']
            all_results.append(results)
        
        # Save consolidated results
        results_path = os.path.join(config['base_training_args']['output_dir'], f"consolidated_results_{int(time.time())}.json")
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nConsolidated results saved to {results_path}")

def run_single_trial(config):
    """Runs a complete training and evaluation pipeline for one configuration."""
    start_time = time.time()

    # 1. Load Model and Tokenizer
    print("Loading model and tokenizer...")
    model, tokenizer = get_model_and_tokenizer(
        config['model_name'],
        config['lora_config'],
        config['training_args']
    )

    # 2. Load and Prepare Data
    print("Loading and preparing datasets...")
    train_dataset, val_dataset = load_and_prepare_datasets(config, tokenizer)

    # 3. Train the Model
    print("Starting model training...")
    trained_model = train_model(config, model, tokenizer, train_dataset, val_dataset)

    # 4. Evaluate the Model
    print("Starting model evaluation...")
    eval_results = run_evaluation(trained_model, tokenizer, config)

    end_time = time.time()
    eval_results['wall_clock_time_seconds'] = end_time - start_time

    # 5. Save results
    output_dir = config['training_args']['output_dir']
    os.makedirs(output_dir, exist_ok=True)
    results_filename = os.path.join(output_dir, "results.json")
    try:
        with open(results_filename, 'w') as f:
            json.dump(eval_results, f, indent=2)
        print(f"Results for this run saved to {results_filename}")
        # Also print to stdout for verification
        print("--- Single Run JSON Result ---")
        print(json.dumps(eval_results, indent=2))
        print("------------------------------")
    except Exception as e:
        print(f"Could not save results to {results_filename}: {e}")
    
    return eval_results

def main():
    parser = argparse.ArgumentParser(description="Run GAPO experiments.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--smoke-test", action='store_true', help="Run a quick smoke test.")
    group.add_argument("--full-experiment", action='store_true', help="Run the full experiment.")
    
    args = parser.parse_args()

    if args.smoke_test:
        config_path = 'config/smoke_test.yaml'
        config = load_config(config_path)
        run_experiment(config, is_smoke_test=True)
    elif args.full_experiment:
        config_path = 'config/full_experiment.yaml'
        config = load_config(config_path)
        run_experiment(config, is_smoke_test=False)

if __name__ == "__main__":
    main()
