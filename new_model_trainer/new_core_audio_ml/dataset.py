import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import os

class AudioDataset(Dataset):
    def __init__(self, dataset_items, sequence_length, sample_rate=48000, num_conditioning_params=0):
        """
        Args:
            dataset_items (list of dict): List of dataset entries. Each dict should have:
                'input_path': str, path to input audio.
                'target_path': str, path to target audio.
                'conditioning_values': list of float, optional, values for conditioning.
                                         Length must match num_conditioning_params.
            sequence_length (int): The length of the audio sequences to return.
            sample_rate (int): The sample rate to resample audio to.
            num_conditioning_params (int): The number of conditioning parameters expected.
                                           If 0, no conditioning is applied.
        """
        self.dataset_items = dataset_items
        self.sequence_length = sequence_length
        self.sample_rate = sample_rate
        self.num_conditioning_params = num_conditioning_params

        self.input_data = []
        self.target_data = []
        self._load_and_preprocess_data()

    def _load_and_preprocess_data(self):
        """
        Loads all audio files and preprocesses them into sequences
        based on the dataset_items structure.
        """
        for item in self.dataset_items:
            in_path = item['input_path']
            tgt_path = item['target_path']
            conditioning_values = item.get('conditioning_values') # Might be None

            if not os.path.exists(in_path):
                print(f"Warning: Input file {in_path} not found. Skipping item.")
                continue
            if not os.path.exists(tgt_path):
                print(f"Warning: Target file {tgt_path} not found. Skipping item.")
                continue

            if self.num_conditioning_params > 0 and (conditioning_values is None or len(conditioning_values) != self.num_conditioning_params):
                print(f"Warning: Item {in_path} has missing or mismatched conditioning_values (expected {self.num_conditioning_params}). Skipping item.")
                continue

            input_wav, sr_in = librosa.load(in_path, sr=self.sample_rate, mono=True)
            target_wav, sr_tgt = librosa.load(tgt_path, sr=self.sample_rate, mono=True)

            min_len = min(len(input_wav), len(target_wav))
            input_wav = input_wav[:min_len]
            target_wav = target_wav[:min_len]

            # Calculate how many full sequences we can get
            num_sequences = len(input_wav) // self.sequence_length

            if num_sequences <= 0:
                print(f"Warning: Audio file pair {in_path}/{tgt_path} is shorter than sequence_length after common trimming. Skipping.")
                continue

            for i in range(num_sequences):
                start_idx = i * self.sequence_length
                end_idx = start_idx + self.sequence_length

                current_audio_sequence = input_wav[start_idx:end_idx]
                current_audio_sequence_expanded = np.expand_dims(current_audio_sequence, axis=1) # [seq_len, 1]

                if self.num_conditioning_params > 0 and conditioning_values is not None:
                    param_arrays = []
                    for p_idx in range(self.num_conditioning_params):
                        param_arrays.append(np.full((self.sequence_length, 1), conditioning_values[p_idx]))

                    # Concatenate audio with conditioning parameters
                    current_input_sequence_final = np.concatenate(
                        [current_audio_sequence_expanded] + param_arrays, axis=1 # Result: [seq_len, 1 + num_params]
                    )
                else: # No conditioning parameters
                    current_input_sequence_final = current_audio_sequence_expanded # Result: [seq_len, 1]

                self.input_data.append(torch.from_numpy(current_input_sequence_final).float())
                self.target_data.append(torch.from_numpy(target_wav[start_idx:end_idx]).float().unsqueeze(-1))

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        if idx >= len(self.input_data): # Should not happen with DataLoader if __len__ is correct
            raise IndexError("Index out of bounds")

        input_sample = self.input_data[idx]
        target_sample = self.target_data[idx]

        return input_sample, target_sample

# Example Usage (for testing purposes)
if __name__ == '__main__':
    import soundfile as sf
    dummy_sr = 48000
    duration = 5
    seq_len_example = 1024

    temp_data_dir = 'temp_data_dataset_cond'
    if not os.path.exists(temp_data_dir):
        os.makedirs(temp_data_dir)

    # Create dummy audio files
    paths = {}
    for i in range(2): # Two sets of files
        paths[f'input{i}'] = os.path.join(temp_data_dir, f'dummy_input{i}.wav')
        paths[f'target{i}'] = os.path.join(temp_data_dir, f'dummy_target{i}.wav')
        if not os.path.exists(paths[f'input{i}']):
            sf.write(paths[f'input{i}'], np.random.randn(dummy_sr * duration).astype(np.float32), dummy_sr)
        if not os.path.exists(paths[f'target{i}']):
            sf.write(paths[f'target{i}'], np.random.randn(dummy_sr * duration).astype(np.float32), dummy_sr)

    print("--- Testing Non-Conditioned (num_conditioning_params=0) ---")
    dataset_items_no_cond = [
        {'input_path': paths['input0'], 'target_path': paths['target0']},
        {'input_path': paths['input1'], 'target_path': paths['target1']},
    ]
    dataset_no_cond = AudioDataset(
        dataset_items=dataset_items_no_cond,
        sequence_length=seq_len_example,
        sample_rate=dummy_sr,
        num_conditioning_params=0
    )
    if len(dataset_no_cond) > 0:
        dataloader_no_cond = DataLoader(dataset_no_cond, batch_size=2, shuffle=True)
        in_batch, tgt_batch = next(iter(dataloader_no_cond))
        print(f"Batch (No Cond): Input shape: {in_batch.shape}, Target shape: {tgt_batch.shape}")
        assert in_batch.shape[-1] == 1 # Feature dim should be 1 (audio only)
    else:
        print("Non-conditioned dataset is empty.")

    print("\n--- Testing Conditioned (num_conditioning_params=2) ---")
    dataset_items_cond = [
        {'input_path': paths['input0'], 'target_path': paths['target0'], 'conditioning_values': [0.1, 0.2]},
        {'input_path': paths['input1'], 'target_path': paths['target1'], 'conditioning_values': [0.3, 0.4]},
        {'input_path': paths['input0'], 'target_path': paths['target0'], 'conditioning_values': [0.5, 0.6]}
    ]
    dataset_cond = AudioDataset(
        dataset_items=dataset_items_cond,
        sequence_length=seq_len_example,
        sample_rate=dummy_sr,
        num_conditioning_params=2
    )
    if len(dataset_cond) > 0:
        dataloader_cond = DataLoader(dataset_cond, batch_size=2, shuffle=True)
        in_batch_cond, tgt_batch_cond = next(iter(dataloader_cond))
        print(f"Batch (Cond): Input shape: {in_batch_cond.shape}, Target shape: {tgt_batch_cond.shape}")
        assert in_batch_cond.shape[-1] == 3 # Feature dim should be 3 (audio + 2 params)
        # Check if conditioning values are different for different items in batch if they came from different dataset_items
        # This requires a more complex check based on how batches are formed from multiple items.
        # For now, shape check is primary.
    else:
        print("Conditioned dataset is empty.")

    print("\nDataset tests complete.")
    # Clean up dummy files (optional)
    # for i in range(2):
    #     if os.path.exists(paths[f'input{i}']): os.remove(paths[f'input{i}'])
    #     if os.path.exists(paths[f'target{i}']): os.remove(paths[f'target{i}'])
    # if os.path.exists(temp_data_dir): os.rmdir(temp_data_dir)
