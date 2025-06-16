import argparse
import json
import os
import glob
import torch
from sklearn.model_selection import train_test_split # For splitting data

# Attempt to import local modules
try:
    from new_core_audio_ml.dataset import AudioDataset
    from new_core_audio_ml.networks import SimpleLSTM
    from new_core_audio_ml.losses import LossWrapper
    from new_core_audio_ml.trainer import Trainer
    from new_core_audio_ml.export_rtneural import export_model_to_rtneural_json
except ImportError as e:
    print(f"Error importing local modules: {e}. Ensure you are in the 'new_model_trainer' directory or have it in PYTHONPATH.")
    # Define placeholders if running in a limited environment (like some subtasks)
    AudioDataset = SimpleLSTM = LossWrapper = Trainer = export_model_to_rtneural_json = None

def load_config(config_path):
    """Loads a JSON configuration file."""
    if not config_path or not os.path.exists(config_path):
        return {}
    with open(config_path, 'r') as f:
        return json.load(f)

def get_audio_files(audio_dir):
    """Gets a list of .wav files from a directory."""
    if not os.path.isdir(audio_dir):
        # Check if it's a single file
        if os.path.isfile(audio_dir) and audio_dir.lower().endswith('.wav'):
            return [audio_dir]
        print(f"Warning: Audio directory {audio_dir} not found or is not a directory.")
        return []

    files = glob.glob(os.path.join(audio_dir, '*.wav'))
    if not files:
        print(f"Warning: No .wav files found in {audio_dir}.")
    return files

def main(args):
    if AudioDataset is None: # Check if imports failed
        print("Core library modules not loaded. Cannot proceed with training.")
        return

    # --- Configuration Loading ---
    config = load_config(args.config_file)

    # Override config with command-line arguments if provided
    config['model_name'] = args.model_name if args.model_name else config.get('model_name', 'default_model')
    config['epochs'] = args.epochs if args.epochs else config.get('epochs', 100)
    config['batch_size'] = args.batch_size if args.batch_size else config.get('batch_size', 32)
    config['learning_rate'] = args.learning_rate if args.learning_rate else config.get('learning_rate', 1e-3)
    config['hidden_size'] = args.hidden_size if args.hidden_size else config.get('hidden_size', 32)
    config['num_layers'] = args.num_layers if args.num_layers else config.get('num_layers', 1)
    config['skip_connection'] = args.skip_connection if args.skip_connection is not None else config.get('skip_connection', True)
    config['sequence_length'] = args.sequence_length if args.sequence_length else config.get('sequence_length', 2048)
    config['sample_rate'] = args.sample_rate if args.sample_rate else config.get('sample_rate', 48000)
    config['device'] = args.device if args.device else config.get('device', 'cpu')
    config['early_stopping_patience'] = args.early_stopping_patience if args.early_stopping_patience else config.get('early_stopping_patience', 10)
    config['output_dir'] = args.output_dir if args.output_dir else config.get('output_dir', 'training_results')

    conditioning_params = None
    if args.conditioning_param1 is not None or args.conditioning_param2 is not None:
        conditioning_params = {
            'param1': args.conditioning_param1 if args.conditioning_param1 is not None else config.get('conditioning_params', {}).get('param1', 0.0),
            'param2': args.conditioning_param2 if args.conditioning_param2 is not None else config.get('conditioning_params', {}).get('param2', 0.0)
        }
    elif 'conditioning_params' in config:
         conditioning_params = config['conditioning_params']

    config['conditioning_params'] = conditioning_params # Store resolved params back in config

    os.makedirs(config['output_dir'], exist_ok=True)
    print(f"Resolved configuration: {json.dumps(config, indent=2)}")

    # --- Device Setup ---
    if config['device'] == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Switching to CPU.")
        config['device'] = 'cpu'
    elif config['device'] == 'mps' and not torch.backends.mps.is_available(): # Corrected MPS check
        print("Warning: MPS requested but not available. Switching to CPU.")
        config['device'] = 'cpu'
    device = torch.device(config['device'])
    print(f"Using device: {device}")

    # --- Data Preparation ---
    input_audio_files_all = get_audio_files(args.input_audio_dir)
    target_audio_files_all = get_audio_files(args.target_audio_dir)

    if not input_audio_files_all or not target_audio_files_all:
        print("Error: Input or target audio files are missing. Exiting.")
        return

    if len(input_audio_files_all) != len(target_audio_files_all):
        print("Warning: Number of input and target audio files do not match. Using common subset based on filename (heuristic).")
        # Simple heuristic: try to match by base names, could be improved
        input_basenames = {os.path.basename(f): f for f in input_audio_files_all}
        target_basenames = {os.path.basename(f): f for f in target_audio_files_all}
        common_basenames = set(input_basenames.keys()) & set(target_basenames.keys())
        if not common_basenames:
            print("Error: No matching input/target file pairs found by basename. Exiting.")
            return
        input_audio_files_all = [input_basenames[b] for b in common_basenames]
        target_audio_files_all = [target_basenames[b] for b in common_basenames]


    # Split data into training and validation sets (e.g., 80/20)
    # Ensure pairs are kept together for splitting
    file_pairs = list(zip(input_audio_files_all, target_audio_files_all))
    if len(file_pairs) < 2 and args.validation_split > 0:
        print("Warning: Not enough audio files for a train/validation split. Using all for training and validation.")
        train_pairs = file_pairs
        val_pairs = file_pairs
    elif args.validation_split == 0:
        print("Validation split is 0. Using all data for training and validation (not recommended for proper eval).")
        train_pairs = file_pairs
        val_pairs = file_pairs
    else:
        train_pairs, val_pairs = train_test_split(file_pairs, test_size=args.validation_split, random_state=42, shuffle=True)

    train_input_paths = [p[0] for p in train_pairs]
    train_target_paths = [p[1] for p in train_pairs]
    val_input_paths = [p[0] for p in val_pairs]
    val_target_paths = [p[1] for p in val_pairs]

    print(f"Training files: {len(train_input_paths)}, Validation files: {len(val_input_paths)}")
    if not train_input_paths:
        print("Error: No training files available after split. Exiting.")
        return

    train_dataset = AudioDataset(
        input_audio_paths=train_input_paths,
        target_audio_paths=train_target_paths,
        sequence_length=config['sequence_length'],
        sample_rate=config['sample_rate'],
        conditioning_params=config['conditioning_params']
    )
    val_dataset = AudioDataset(
        input_audio_paths=val_input_paths,
        target_audio_paths=val_target_paths,
        sequence_length=config['sequence_length'],
        sample_rate=config['sample_rate'],
        conditioning_params=config['conditioning_params']
    )

    if len(train_dataset) == 0:
        print("Error: Training dataset is empty. Check audio files and sequence length. Exiting.")
        return
    if len(val_dataset) == 0 and args.validation_split > 0 : # Only critical if a split was expected
        print("Warning: Validation dataset is empty. Proceeding with training but validation will be skipped or may fail.")
        # Trainer might need to handle an empty val_loader gracefully or we could skip validation

    # --- Model, Loss, Optimizer Setup ---
    model_input_size = 3 if config['conditioning_params'] else 1
    model = SimpleLSTM(
        input_size=model_input_size,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        output_size=1, # Assuming single audio output
        skip_connection=config['skip_connection']
    )

    loss_configs = config.get('loss_functions', {'ESR': {'weight': 0.8}, 'DC': {'weight': 0.2}})
    pre_emph_config = config.get('pre_emphasis', {'apply': True, 'type': 'aw'}) # A-weighting default

    loss_fn = LossWrapper(
        loss_configs=loss_configs,
        pre_emph_config=pre_emph_config,
        sample_rate=config['sample_rate']
    )

    # --- Trainer Initialization and Training ---
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset, # Trainer should handle if val_dataset is empty
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
    trainer.train(resume_from_latest=args.resume) # Pass resume flag

    # --- Export Model ---
    print("Training complete. Exporting the best model to RTNeural format...")
    best_model_path = os.path.join(config['output_dir'], 'checkpoints', config['model_name'], f"{config['model_name']}_best.pth")

    # Load the best model's state dict
    # The Trainer already loads the best model if load_best_checkpoint_at_end=True (default)
    # So, trainer.model should be the best model here.

    if not os.path.exists(best_model_path):
        print(f"Warning: Best model checkpoint {best_model_path} not found. Cannot export.")
    else:
        # Re-instantiate model and load best state to be sure, or trust trainer.model
        # For safety, explicitly load the best model for export
        export_model_instance = SimpleLSTM(
            input_size=model_input_size,
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            output_size=1,
            skip_connection=config['skip_connection']
        )
        # Load checkpoint using the model's class method
        try:
            # Assuming the save_pytorch_model saved a dict with 'state_dict' and 'model_config'
            checkpoint = torch.load(best_model_path, map_location='cpu') # Load to CPU first
            if 'state_dict' in checkpoint:
                 export_model_instance.load_state_dict(checkpoint['state_dict'])
            elif 'model_config' in checkpoint and 'state_dict' in checkpoint : # Saved by new SimpleLSTM
                 export_model_instance.load_state_dict(checkpoint['state_dict'])
            else: # Directly a state_dict
                 export_model_instance.load_state_dict(checkpoint)
            export_model_instance.eval() # Set to evaluation mode
            print(f"Successfully loaded best model from {best_model_path} for export.")

            output_aidax_filename = os.path.join(config['output_dir'], f"{config['model_name']}.aidax")

            # Gather metadata for export
            # Try to get ESR from training_stats.json if it exists
            esr_val_best = None
            stats_path = os.path.join(config['output_dir'], 'logs', config['model_name'], 'training_stats.json')
            if os.path.exists(stats_path):
                with open(stats_path, 'r') as f_stats:
                    training_stats_data = json.load(f_stats)
                    if training_stats_data['epoch_val_loss']: # Check if not empty
                         esr_val_best = min(training_stats_data['epoch_val_loss']) # Assuming val_loss is primary metric

            # Dummy input/output batch examples (replace with actual from dataset if desired)
            # For now, keeping them as None or simple placeholders
            input_batch_ex = [[0.0] * config['sequence_length']] if model_input_size == 1 else [[ [0.0]*model_input_size ]*config['sequence_length']]
            output_batch_ex = [[0.0] * config['sequence_length']]

            export_model_to_rtneural_json(
                pytorch_model=export_model_instance,
                output_aidax_path=output_aidax_filename,
                sample_rate=config['sample_rate'],
                model_name=config['model_name'],
                model_version=config.get('model_version', "1.0.0"),
                esr_val=esr_val_best, # Pass the best validation loss as ESR
                input_batch_example=input_batch_ex, # Placeholder
                output_batch_example=output_batch_ex # Placeholder
            )
        except Exception as e:
            print(f"Error during model export: {e}")

    print("Main script finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train an audio model and export to RTNeural format.")

    # Paths
    parser.add_argument('--input_audio_dir', type=str, required=True, help="Directory containing input .wav files (or path to a single file).")
    parser.add_argument('--target_audio_dir', type=str, required=True, help="Directory containing target .wav files (or path to a single file).")
    parser.add_argument('--output_dir', type=str, default='training_results', help="Directory to save models, logs, and exported files.")
    parser.add_argument('--config_file', type=str, default=None, help="Path to a JSON training configuration file.")
    parser.add_argument('--resume', action='store_true', help="Resume training from the latest checkpoint if available.")


    # Model Config (can be overridden by config_file)
    parser.add_argument('--model_name', type=str, default=None, help="Name for the model and output files.")
    parser.add_argument('--hidden_size', type=int, default=None, help="LSTM hidden size.")
    parser.add_argument('--num_layers', type=int, default=None, help="Number of LSTM layers.")
    parser.add_argument('--skip_connection', type=lambda x: (str(x).lower() == 'true'), default=None, help="Enable skip connection (True/False).")

    # Training Params (can be overridden by config_file)
    parser.add_argument('--epochs', type=int, default=None, help="Number of training epochs.")
    parser.add_argument('--batch_size', type=int, default=None, help="Batch size.")
    parser.add_argument('--learning_rate', type=float, default=None, help="Initial learning rate.")
    parser.add_argument('--sequence_length', type=int, default=None, help="Length of audio sequences for training.")
    parser.add_argument('--sample_rate', type=int, default=None, help="Target sample rate for audio processing.")
    parser.add_argument('--device', type=str, default=None, help="Device to use for training ('cpu', 'cuda', 'mps').")
    parser.add_argument('--early_stopping_patience', type=int, default=None, help="Patience for early stopping.")
    parser.add_argument('--validation_split', type=float, default=0.2, help="Fraction of data to use for validation (0 to disable, e.g., 0.2 for 20%).")


    # Conditioning Params (can be overridden by config_file)
    parser.add_argument('--conditioning_param1', type=float, default=None, help="Value for the first conditioning parameter.")
    parser.add_argument('--conditioning_param2', type=float, default=None, help="Value for the second conditioning parameter.")

    args = parser.parse_args()
    main(args)
