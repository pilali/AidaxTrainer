import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader # Added for example
import os
import time
import json # For saving training stats

# Attempt to import from local modules, handle if not found during subtask execution
try:
    from .dataset import AudioDataset
    from .networks import SimpleLSTM
    from .losses import LossWrapper
except ImportError:
    # This fallback is for subtask execution where the environment might be tricky
    print("Warning: Could not import local modules (dataset, networks, losses). Using placeholders if in subtask.")
    AudioDataset = None
    SimpleLSTM = None
    LossWrapper = None


class Trainer:
    def __init__(self, model, train_dataset, val_dataset, loss_fn,
                 optimizer_name='AdamW', learning_rate=1e-3, scheduler_name='ReduceLROnPlateau',
                 device='cpu', save_dir='results', model_name='model',
                 epochs=100, batch_size=32, early_stopping_patience=10,
                 log_interval=10, grad_clip_value=None):
        """
        Trainer class for audio models.

        Args:
            model (torch.nn.Module): The neural network model.
            train_dataset (Dataset): Training dataset.
            val_dataset (Dataset): Validation dataset.
            loss_fn (torch.nn.Module): Loss function.
            optimizer_name (str): Name of the optimizer (e.g., 'Adam', 'AdamW').
            learning_rate (float): Learning rate.
            scheduler_name (str): Name of the LR scheduler (e.g., 'ReduceLROnPlateau', 'StepLR').
            device (str): Device to train on ('cpu', 'cuda', 'mps').
            save_dir (str): Directory to save models and logs.
            model_name (str): Base name for saved model files.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training and validation.
            early_stopping_patience (int): Patience for early stopping.
            log_interval (int): Interval (in batches) for logging training loss.
            grad_clip_value (float, optional): Value for gradient clipping. None to disable.
        """
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.loss_fn = loss_fn.to(device)
        self.device = device
        self.save_dir = save_dir
        self.model_name = model_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        self.log_interval = log_interval
        self.grad_clip_value = grad_clip_value

        # Optimizer
        if optimizer_name.lower() == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        elif optimizer_name.lower() == 'adamw':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        # Scheduler
        if scheduler_name.lower() == 'reducelronplateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=max(2, early_stopping_patience // 2), factor=0.5)
        elif scheduler_name.lower() == 'steplr':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.1)
        else:
            # Default to no scheduler or raise error
            self.scheduler = None
            print(f"Warning: Scheduler '{scheduler_name}' not explicitly supported or 'None'. No LR scheduling will be used unless ReduceLROnPlateau is default.")


        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True if device=='cuda' else False)
        self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True if device=='cuda' else False)

        self.writer = SummaryWriter(log_dir=os.path.join(self.save_dir, 'logs', self.model_name))
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        self.training_stats = {
            'epoch_train_loss': [],
            'epoch_val_loss': [],
            'epoch_lr': []
        }

        os.makedirs(os.path.join(self.save_dir, 'checkpoints', self.model_name), exist_ok=True)

    def _save_checkpoint(self, epoch, is_best=False):
        checkpoint_name = f"{self.model_name}_latest.pth"
        if is_best:
            checkpoint_name = f"{self.model_name}_best.pth"

        checkpoint_path = os.path.join(self.save_dir, 'checkpoints', self.model_name, checkpoint_name)

        # Use the model's save method if available, otherwise save state_dict
        if hasattr(self.model, 'save_pytorch_model') and callable(getattr(self.model, 'save_pytorch_model')):
            self.model.save_pytorch_model(checkpoint_path)
        else:
            torch.save(self.model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    def _load_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint file {checkpoint_path} not found. Starting from scratch.")
            return False

        # Use model's load method if available
        if hasattr(self.model, 'load_pytorch_model') and callable(getattr(self.model, 'load_pytorch_model')):
            try:
                self.model = self.model.load_pytorch_model(checkpoint_path) # Assumes load_pytorch_model is a classmethod or handles instance
                self.model.to(self.device) # Ensure model is on correct device after loading
                print(f"Loaded checkpoint from {checkpoint_path} using model's method.")
                return True
            except Exception as e:
                print(f"Error loading checkpoint with model's method: {e}. Attempting to load state_dict.")
                # Fallback to loading state_dict if model's method fails or isn't suitable
                try:
                    self.model.load_state_dict(torch.load(checkpoint_path)['state_dict']) # Assuming save_pytorch_model saves a dict with 'state_dict'
                    self.model.to(self.device)
                    print(f"Loaded state_dict from {checkpoint_path}.")
                    return True
                except Exception as e_sd:
                    print(f"Error loading state_dict: {e_sd}. Starting from scratch.")
                    return False
        else: # Fallback to loading state_dict directly
            try:
                self.model.load_state_dict(torch.load(checkpoint_path))
                self.model.to(self.device)
                print(f"Loaded state_dict directly from {checkpoint_path}.")
                return True
            except Exception as e:
                print(f"Error loading state_dict directly: {e}. Starting from scratch.")
                return False


    def train_epoch(self, epoch):
        self.model.train()
        epoch_loss = 0.0
        num_batches = len(self.train_loader)

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            # Reset hidden state for LSTM-like models at the beginning of each batch
            if hasattr(self.model, 'reset_hidden'):
                self.model.reset_hidden(batch_size=inputs.size(0), device=self.device)

            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()

            if self.grad_clip_value:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_value)

            self.optimizer.step()

            epoch_loss += loss.item()

            if (batch_idx + 1) % self.log_interval == 0:
                print(f"Epoch {epoch+1}/{self.epochs}, Batch {batch_idx+1}/{num_batches}, Batch Loss: {loss.item():.4f}")
                self.writer.add_scalar('Loss/Train_Batch', loss.item(), epoch * num_batches + batch_idx)

            # Detach hidden state for LSTM-like models to prevent backpropagating through entire history
            if hasattr(self.model, 'detach_hidden'):
                self.model.detach_hidden()

        avg_epoch_loss = epoch_loss / num_batches
        return avg_epoch_loss

    def validate_epoch(self, epoch):
        self.model.eval()
        epoch_val_loss = 0.0
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.val_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                if hasattr(self.model, 'reset_hidden'):
                    self.model.reset_hidden(batch_size=inputs.size(0), device=self.device)

                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                epoch_val_loss += loss.item()

                if hasattr(self.model, 'detach_hidden'):
                    self.model.detach_hidden()

        avg_epoch_val_loss = epoch_val_loss / num_batches
        return avg_epoch_val_loss

    def train(self, load_best_checkpoint_at_end=True, resume_from_latest=False):
        if resume_from_latest:
            latest_checkpoint_path = os.path.join(self.save_dir, 'checkpoints', self.model_name, f"{self.model_name}_latest.pth")
            self._load_checkpoint(latest_checkpoint_path) # Implement loading logic if needed for epoch, optimizer state etc.

        start_time = time.time()
        print(f"Starting training for {self.epochs} epochs on device: {self.device}")

        for epoch in range(self.epochs):
            epoch_start_time = time.time()

            train_loss = self.train_epoch(epoch)
            val_loss = self.validate_epoch(epoch)

            epoch_duration = time.time() - epoch_start_time
            current_lr = self.optimizer.param_groups[0]['lr']

            print(f"Epoch {epoch+1}/{self.epochs} - Duration: {epoch_duration:.2f}s - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - LR: {current_lr:.2e}")

            self.writer.add_scalar('Loss/Train_Epoch', train_loss, epoch)
            self.writer.add_scalar('Loss/Validation_Epoch', val_loss, epoch)
            self.writer.add_scalar('LearningRate', current_lr, epoch)

            self.training_stats['epoch_train_loss'].append(train_loss)
            self.training_stats['epoch_val_loss'].append(val_loss)
            self.training_stats['epoch_lr'].append(current_lr)


            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                elif not isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau): # For StepLR etc.
                    self.scheduler.step()


            self._save_checkpoint(epoch, is_best=False) # Save latest model

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch+1} due to no improvement in validation loss for {self.early_stopping_patience} epochs.")
                    break

        total_training_time = time.time() - start_time
        print(f"Training finished. Total time: {total_training_time // 60:.0f}m {total_training_time % 60:.0f}s")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

        if load_best_checkpoint_at_end:
            best_checkpoint_path = os.path.join(self.save_dir, 'checkpoints', self.model_name, f"{self.model_name}_best.pth")
            if self._load_checkpoint(best_checkpoint_path):
                 print(f"Loaded best model weights from {best_checkpoint_path} for potential export.")
            else:
                print(f"Warning: Could not load best model from {best_checkpoint_path}.")

        self.writer.close()

        # Save training stats
        stats_path = os.path.join(self.save_dir, 'logs', self.model_name, 'training_stats.json')
        with open(stats_path, 'w') as f:
            json.dump(self.training_stats, f, indent=4)
        print(f"Training statistics saved to {stats_path}")


# Example Usage (for testing purposes, may not run in subtask due to missing modules)
if __name__ == '__main__':
    print("Trainer Example Usage:")
    # This example requires dummy versions of Dataset, Model, Loss which might not be available
    # if the imports at the top failed.
    if AudioDataset is None or SimpleLSTM is None or LossWrapper is None:
        print("Cannot run Trainer example because core modules (Dataset, Network, Loss) are not available.")
    else:
        # Create dummy data and model
        # These paths won't exist in the subtask but are for local testing illustration
        dummy_input_file = './dummy_train_input.wav'
        dummy_target_file = './dummy_train_target.wav'

        # Create dummy wav files if they don't exist (requires soundfile)
        try:
            import soundfile as sf
            import numpy as np
            if not os.path.exists(dummy_input_file):
                sf.write(dummy_input_file, np.random.randn(48000 * 2), 48000) # 2 sec audio
            if not os.path.exists(dummy_target_file):
                sf.write(dummy_target_file, np.random.randn(48000 * 2), 48000)
        except ImportError:
            print("Skipping dummy file creation: soundfile or numpy not available.")
            # Fallback: if files can't be created, dataset instantiation will fail later.
            # For robust testing, these files should be pre-existing or created reliably.
            dummy_input_file = None


        if dummy_input_file: # Proceed only if dummy files could be handled
            train_ds = AudioDataset(
                input_audio_paths=[dummy_input_file],
                target_audio_paths=[dummy_target_file],
                sequence_length=1024
            )
            val_ds = AudioDataset(
                input_audio_paths=[dummy_input_file],
                target_audio_paths=[dummy_target_file],
                sequence_length=1024
            )

            if len(train_ds) == 0 or len(val_ds) == 0:
                print("Dummy dataset is empty. Ensure dummy audio files are valid and long enough.")
            else:
                model_instance = SimpleLSTM(input_size=1, hidden_size=16) # Non-conditioned

                loss_configs_example = {'ESR': {'weight': 1.0}}
                loss_fn_instance = LossWrapper(loss_configs=loss_configs_example)

                device_to_use = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
                print(f"Using device: {device_to_use}")

                trainer_instance = Trainer(
                    model=model_instance,
                    train_dataset=train_ds,
                    val_dataset=val_ds,
                    loss_fn=loss_fn_instance,
                    device=device_to_use,
                    save_dir='temp_trainer_results',
                    model_name='dummy_lstm_model',
                    epochs=2, # Keep epochs low for a quick test
                    batch_size=2,
                    early_stopping_patience=3,
                    log_interval=1
                )

                print("Starting dummy training...")
                trainer_instance.train(load_best_checkpoint_at_end=False) # Don't load best to avoid issues if save failed
                print("Dummy training complete.")

                # Basic check if log files were created
                log_dir_path = os.path.join('temp_trainer_results', 'logs', 'dummy_lstm_model')
                if os.path.exists(log_dir_path) and any(os.scandir(log_dir_path)): # Check if dir exists and is not empty
                    print(f"TensorBoard logs should be in: {log_dir_path}")
                else:
                    print(f"TensorBoard log directory not found or empty: {log_dir_path}")

                # Clean up dummy files and dirs (optional)
                # if os.path.exists(dummy_input_file): os.remove(dummy_input_file)
                # if os.path.exists(dummy_target_file): os.remove(dummy_target_file)
                # import shutil
                # if os.path.exists('temp_trainer_results'): shutil.rmtree('temp_trainer_results')
        else:
            print("Skipping trainer example as dummy audio files could not be prepared.")
