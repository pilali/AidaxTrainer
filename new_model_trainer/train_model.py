import argparse
import json
import os
import glob
import torch
from sklearn.model_selection import train_test_split

try:
    from new_core_audio_ml.dataset import AudioDataset
    from new_core_audio_ml.networks import SimpleLSTM
    from new_core_audio_ml.losses import LossWrapper
    from new_core_audio_ml.trainer import Trainer
    from new_core_audio_ml.export_rtneural import export_model_to_rtneural_json
except ImportError as e:
    print(f"Error importing local modules: {e}. Ensure you are in the 'new_model_trainer' directory or have it in PYTHONPATH.")
    AudioDataset = SimpleLSTM = LossWrapper = Trainer = export_model_to_rtneural_json = None

def load_config(config_path):
    if not config_path or not os.path.exists(config_path):
        return {}
    with open(config_path, 'r') as f:
        return json.load(f)

def main(args):
    if AudioDataset is None:
        print("Core library modules not loaded. Cannot proceed.")
        return

    config = load_config(args.config_file)

    # --- Configuration Merging (CLI overrides config file) ---
    config['model_name'] = args.model_name if args.model_name is not None else config.get('model_name', 'default_model')
    config['output_dir'] = args.output_dir if args.output_dir is not None else config.get('output_dir', 'training_results')

    # Training parameters
    config['epochs'] = args.epochs if args.epochs is not None else config.get('epochs', 100)
    config['batch_size'] = args.batch_size if args.batch_size is not None else config.get('batch_size', 32)
    config['learning_rate'] = args.learning_rate if args.learning_rate is not None else config.get('learning_rate', 1e-3)
    config['device'] = args.device if args.device is not None else config.get('device', 'cpu')
    config['early_stopping_patience'] = args.early_stopping_patience if args.early_stopping_patience is not None else config.get('early_stopping_patience', 10)
    config['validation_split'] = args.validation_split if args.validation_split is not None else config.get('validation_split', 0.2)
    config['resume_from_latest'] = args.resume if args.resume is not None else config.get('resume_from_latest', False)

    # Model architecture parameters
    config['hidden_size'] = args.hidden_size if args.hidden_size is not None else config.get('hidden_size', 32)
    config['num_layers'] = args.num_layers if args.num_layers is not None else config.get('num_layers', 1) # For LSTM layers within SimpleLSTM
    config['skip_connection'] = args.skip_connection if args.skip_connection is not None else config.get('skip_connection', True)

    # Data parameters
    config['sequence_length'] = args.sequence_length if args.sequence_length is not None else config.get('sequence_length', 2048)
    config['sample_rate'] = args.sample_rate if args.sample_rate is not None else config.get('sample_rate', 48000)

    # Dataset items - CLI cannot easily override this complex structure, primarily from config file
    # The args.input_audio_dir and args.target_audio_dir are for simple, non-conditioned, single/multi-file cases without complex metadata.
    # For conditioned models, the config file's "dataset_items" structure is preferred.

    dataset_items_from_config = config.get('dataset_items', [])
    num_conditioning_params_from_config = config.get('num_conditioning_params', 0)

    # If CLI audio dirs are given AND dataset_items is empty in config, create a basic dataset_item list
    if args.input_audio_dir and args.target_audio_dir and not dataset_items_from_config:
        print("Using input/target audio directories from command line for dataset construction.")
        input_files = sorted(glob.glob(os.path.join(args.input_audio_dir, '*.wav')))
        target_files = sorted(glob.glob(os.path.join(args.target_audio_dir, '*.wav')))

        if not input_files or not target_files:
            print(f"Error: No WAV files found in --input_audio_dir '{args.input_audio_dir}' or --target_audio_dir '{args.target_audio_dir}'.")
            return

        # Simple pairing (assuming same number of files and corresponding order)
        # More sophisticated pairing might be needed for complex cases.
        if len(input_files) != len(target_files):
            print(f"Warning: Mismatch in number of input ({len(input_files)}) and target ({len(target_files)}) files. Attempting to pair based on sorted order up to shortest list.")

        dataset_items_cli = []
        min_len = min(len(input_files), len(target_files))
        for i in range(min_len):
            dataset_items_cli.append({
                "input_path": input_files[i],
                "target_path": target_files[i]
                # No conditioning_values from CLI in this simple mode
            })
        config['dataset_items'] = dataset_items_cli
        config['num_conditioning_params'] = 0 # CLI mode implies no conditioning unless config specifies otherwise
        print(f"Constructed {len(dataset_items_cli)} dataset items from CLI paths.")
    elif dataset_items_from_config:
        config['dataset_items'] = dataset_items_from_config
        # Infer num_conditioning_params if not explicitly set, from first item that has conditioning_values
        if num_conditioning_params_from_config == 0: # If not set in config, try to infer
            for item in config['dataset_items']:
                if item.get('conditioning_values') is not None:
                    num_conditioning_params_from_config = len(item['conditioning_values'])
                    break
        config['num_conditioning_params'] = num_conditioning_params_from_config
        print(f"Using dataset_items from config file. Inferred/Set num_conditioning_params: {config['num_conditioning_params']}")
    else:
        print("Error: No dataset_items found in config and --input_audio_dir/--target_audio_dir not provided or empty. Cannot proceed.")
        return

    os.makedirs(config['output_dir'], exist_ok=True)
    print(f"Resolved configuration: {json.dumps(config, indent=2)}")

    # --- Device Setup ---
    if config['device'] == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Switching to CPU.")
        config['device'] = 'cpu'
    elif config['device'] == 'mps' and not torch.backends.mps.is_available():
        print("Warning: MPS requested but not available. Switching to CPU.")
        config['device'] = 'cpu'
    device = torch.device(config['device'])
    print(f"Using device: {device}")

    # --- Data Preparation ---
    all_dataset_items = config['dataset_items']
    if not all_dataset_items:
        print("Error: dataset_items is empty. No data to train on. Exiting.")
        return

    train_items, val_items = train_test_split(all_dataset_items, test_size=config['validation_split'], random_state=42, shuffle=True)

    if not train_items:
        print("Error: No training items after split. Check dataset and validation_split. Exiting.")
        return

    print(f"Total dataset items: {len(all_dataset_items)}. Training items: {len(train_items)}, Validation items: {len(val_items)}")

    train_dataset = AudioDataset(
        dataset_items=train_items,
        sequence_length=config['sequence_length'],
        sample_rate=config['sample_rate'],
        num_conditioning_params=config['num_conditioning_params']
    )
    # Create validation dataset only if there are validation items
    val_dataset = None
    if val_items:
        val_dataset = AudioDataset(
            dataset_items=val_items,
            sequence_length=config['sequence_length'],
            sample_rate=config['sample_rate'],
            num_conditioning_params=config['num_conditioning_params']
        )
        if len(val_dataset) == 0 : # If val_items existed but resulted in empty dataset
            print("Warning: Validation dataset is empty after processing validation items. No validation will be performed.")
            val_dataset = None # Explicitly set to None
    else:
        print("No items for validation set based on split. Validation will be skipped.")


    if len(train_dataset) == 0:
        print("Error: Training dataset is empty after processing. Check audio files, paths, and sequence length. Exiting.")
        return

    # --- Model, Loss, Optimizer Setup ---
    model_input_size = 1 + config['num_conditioning_params']
    model = SimpleLSTM(
        input_size=model_input_size,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        output_size=1,
        skip_connection=config['skip_connection']
    )

    loss_configs = config.get('loss_functions', {'ESR': {'weight': 0.8}, 'DC': {'weight': 0.2}})
    pre_emph_config = config.get('pre_emphasis', {'apply': True, 'type': 'aw'})

    loss_fn = LossWrapper(
        loss_configs=loss_configs,
        pre_emph_config=pre_emph_config,
        sample_rate=config['sample_rate']
    )

    # --- Trainer Initialization and Training ---
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset, # Trainer's train method should handle val_dataset being None
        loss_fn=loss_fn,
        optimizer_name=config.get('optimizer', 'AdamW'),
        learning_rate=config['learning_rate'],
        scheduler_name=config.get('scheduler', 'ReduceLROnPlateau'),
        device=device,
        save_dir=config['output_dir'],
        model_name=config['model_name'],
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        early_stopping_patience=config['early_stopping_patience'],
        log_interval=config.get('log_interval', 10),
        grad_clip_value=config.get('grad_clip_value', None)
    )

    print(f"Starting training for model: {config['model_name']}")
    trainer.train(resume_from_latest=config['resume_from_latest'])

    # --- Export Model ---
    print("Training complete. Exporting the best model to RTNeural format...")
    best_model_checkpoint_path = os.path.join(config['output_dir'], 'checkpoints', config['model_name'], f"{config['model_name']}_best.pth")

    if not os.path.exists(best_model_checkpoint_path):
        print(f"Warning: Best model checkpoint {best_model_checkpoint_path} not found. Attempting to use latest if available.")
        best_model_checkpoint_path = os.path.join(config['output_dir'], 'checkpoints', config['model_name'], f"{config['model_name']}_latest.pth")
        if not os.path.exists(best_model_checkpoint_path):
            print(f"Error: No model checkpoint found at {best_model_checkpoint_path} or as '_latest.pth'. Cannot export.")
            return # Exit if no model can be loaded

    # Use the model instance from the trainer, which should be the best one if load_best_checkpoint_at_end was True (default)
    # Or, explicitly load the state dict into a fresh model instance for export
    export_model_instance = SimpleLSTM(
        input_size=model_input_size,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        output_size=1,
        skip_connection=config['skip_connection']
    )
    try:
        checkpoint = torch.load(best_model_checkpoint_path, map_location='cpu')
        # Handle different checkpoint formats (direct state_dict vs dict with 'state_dict')
        if 'state_dict' in checkpoint:
            export_model_instance.load_state_dict(checkpoint['state_dict'])
        elif 'model_config' in checkpoint and 'state_dict' in checkpoint: # Saved by new SimpleLSTM
            export_model_instance.load_state_dict(checkpoint['state_dict'])
        else: # Assume it's a direct state_dict
            export_model_instance.load_state_dict(checkpoint)
        export_model_instance.eval()
        print(f"Successfully loaded model from {best_model_checkpoint_path} for export.")

        output_aidax_filename = os.path.join(config['output_dir'], f"{config['model_name']}.aidax")

        esr_val_best = trainer.best_val_loss if trainer.best_val_loss != float('inf') else None

        input_batch_ex_data = [[0.0] * config['sequence_length']]
        if model_input_size > 1: # If conditioned
            input_batch_ex_data = [[ [0.0]*model_input_size ]*config['sequence_length']]


        export_model_to_rtneural_json(
            pytorch_model=export_model_instance,
            output_aidax_path=output_aidax_filename,
            sample_rate=config['sample_rate'],
            model_name=config['model_name'],
            model_version=config.get('model_version', "1.0.0"),
            esr_val=esr_val_best,
            input_batch_example=input_batch_ex_data,
            output_batch_example=[[0.0] * config['sequence_length']]
        )
    except Exception as e:
        print(f"Error during model export: {e}")

    print("Main script finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train an audio model and export to RTNeural format.")

    # Paths - Keep these for simple non-conditioned cases or if config file is not used
    parser.add_argument('--input_audio_dir', type=str, default=None, help="Directory containing input .wav files (for simple, non-conditioned cases if not using dataset_items in config).")
    parser.add_argument('--target_audio_dir', type=str, default=None, help="Directory containing target .wav files (for simple, non-conditioned cases).")

    parser.add_argument('--output_dir', type=str, default=None, help="Directory to save models, logs, and exported files.")
    parser.add_argument('--config_file', type=str, default=None, help="Path to a JSON training configuration file.")
    parser.add_argument('--resume', action='store_true', default=None, help="Resume training from the latest checkpoint if available.")

    # Model Config
    parser.add_argument('--model_name', type=str, default=None, help="Name for the model and output files.")
    parser.add_argument('--hidden_size', type=int, default=None, help="LSTM hidden size.")
    parser.add_argument('--num_layers', type=int, default=None, help="Number of LSTM layers.")
    parser.add_argument('--skip_connection', type=lambda x: (str(x).lower() == 'true'), default=None, help="Enable skip connection (True/False).")

    # Training Params
    parser.add_argument('--epochs', type=int, default=None, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=None, help="Batch size.")
    parser.add_argument('--learning_rate', type=float, default=None, help="Initial learning rate.")
    parser.add_argument('--sequence_length', type=int, default=None, help="Length of audio sequences for training.")
    parser.add_argument('--sample_rate', type=int, default=None, help="Target sample rate for audio processing.")
    parser.add_argument('--device', type=str, default=None, help="Device to use for training ('cpu', 'cuda', 'mps').")
    parser.add_argument('--early_stopping_patience', type=int, default=None, help="Patience for early stopping.")
    parser.add_argument('--validation_split', type=float, default=None, help="Fraction of data to use for validation (0 to disable, e.g., 0.2 for 20%).")

    # num_conditioning_params is now primarily read from config, not CLI

    args = parser.parse_args()
    main(args)
