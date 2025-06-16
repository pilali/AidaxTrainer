import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import os

class AudioDataset(Dataset):
    def __init__(self, input_audio_paths, target_audio_paths, sequence_length, sample_rate=48000, conditioning_params=None):
        """
        Args:
            input_audio_paths (list of str): List of paths to input audio files.
            target_audio_paths (list of str): List of paths to target audio files.
            sequence_length (int): The length of the audio sequences to return.
            sample_rate (int): The sample rate to resample audio to.
            conditioning_params (dict, optional): Dictionary of conditioning parameters.
                                                  Example: {'param1': 0.5, 'param2': 0.2}
                                                  If None, model is not conditioned.
        """
        self.input_audio_paths = input_audio_paths
        self.target_audio_paths = target_audio_paths
        self.sequence_length = sequence_length
        self.sample_rate = sample_rate
        self.conditioning_params = conditioning_params

        self.input_data = []
        self.target_data = []
        self._load_and_preprocess_data()

    def _load_and_preprocess_data(self):
        """
        Loads all audio files and preprocesses them into sequences.
        This is a basic implementation and might be memory-intensive for large datasets.
        Consider lazy loading or more advanced chunking for larger datasets.
        """
        for in_path, tgt_path in zip(self.input_audio_paths, self.target_audio_paths):
            if not os.path.exists(in_path):
                print(f"Warning: Input file {in_path} not found. Skipping.")
                continue
            if not os.path.exists(tgt_path):
                print(f"Warning: Target file {tgt_path} not found. Skipping.")
                continue

            input_wav, sr_in = librosa.load(in_path, sr=self.sample_rate, mono=True)
            target_wav, sr_tgt = librosa.load(tgt_path, sr=self.sample_rate, mono=True)

            # Ensure same length, trim longer to shorter
            min_len = min(len(input_wav), len(target_wav))
            input_wav = input_wav[:min_len]
            target_wav = target_wav[:min_len]

            # Create sequences
            num_sequences = (len(input_wav) - self.sequence_length) // self.sequence_length
            if num_sequences <= 0: # Or handle this by padding if sequence_length is too large
                print(f"Warning: Audio file {in_path} is shorter than sequence_length. Skipping.")
                continue

            for i in range(num_sequences):
                start_idx = i * self.sequence_length
                end_idx = start_idx + self.sequence_length

                current_input_sequence = input_wav[start_idx:end_idx]

                if self.conditioning_params:
                    # Expand dimensions to [sequence_length, 1] for audio
                    current_input_sequence = np.expand_dims(current_input_sequence, axis=1)

                    # Create conditioning parameter arrays
                    param1_array = np.full((self.sequence_length, 1), self.conditioning_params.get('param1', 0.0))
                    param2_array = np.full((self.sequence_length, 1), self.conditioning_params.get('param2', 0.0))

                    # Concatenate audio with conditioning parameters: [sequence_length, 3]
                    current_input_sequence = np.concatenate(
                        (current_input_sequence, param1_array, param2_array), axis=1
                    )
                else:
                    # If not conditioned, ensure input is [sequence_length, 1] for consistency
                     current_input_sequence = np.expand_dims(current_input_sequence, axis=1)


                self.input_data.append(torch.from_numpy(current_input_sequence).float())
                self.target_data.append(torch.from_numpy(target_wav[start_idx:end_idx]).float().unsqueeze(-1))


    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        if idx >= len(self.input_data):
            raise IndexError("Index out of bounds")

        input_sample = self.input_data[idx]
        target_sample = self.target_data[idx]

        return input_sample, target_sample

# Example Usage (for testing purposes)
if __name__ == '__main__':
    # Create dummy audio files for testing
    import soundfile as sf
    dummy_sr = 48000
    duration = 5 # seconds

    # Create 'temp_data' directory if it doesn't exist
    temp_data_dir = 'temp_data'
    if not os.path.exists(temp_data_dir):
        os.makedirs(temp_data_dir)

    dummy_input_path = os.path.join(temp_data_dir, 'dummy_input.wav')
    dummy_target_path = os.path.join(temp_data_dir, 'dummy_target.wav')

    # Check if dummy files already exist to avoid recreating them unnecessarily
    if not os.path.exists(dummy_input_path):
        dummy_input_signal = np.random.randn(dummy_sr * duration).astype(np.float32)
        sf.write(dummy_input_path, dummy_input_signal, dummy_sr)
        print(f"Created dummy input file: {dummy_input_path}")

    if not os.path.exists(dummy_target_path):
        dummy_target_signal = np.random.randn(dummy_sr * duration).astype(np.float32)
        sf.write(dummy_target_path, dummy_target_signal, dummy_sr)
        print(f"Created dummy target file: {dummy_target_path}")

    print(f"Loading dummy files for dataset: {dummy_input_path}, {dummy_target_path}")

    # Test non-conditioned model
    print("\nTesting Non-Conditioned Dataset:")
    dataset_no_cond = AudioDataset(
        input_audio_paths=[dummy_input_path],
        target_audio_paths=[dummy_target_path],
        sequence_length=1024,
        sample_rate=dummy_sr
    )
    if len(dataset_no_cond) > 0:
        dataloader_no_cond = DataLoader(dataset_no_cond, batch_size=4, shuffle=True)
        for i, (in_batch, tgt_batch) in enumerate(dataloader_no_cond):
            print(f"Batch {i+1} (No Cond): Input shape: {in_batch.shape}, Target shape: {tgt_batch.shape}")
            # Expect Input shape: [batch_size, sequence_length, 1]
            # Expect Target shape: [batch_size, sequence_length, 1]
            assert in_batch.ndim == 3 and in_batch.shape[2] == 1
            assert tgt_batch.ndim == 3 and tgt_batch.shape[2] == 1
            break
    else:
        print("Non-conditioned dataset is empty. Check audio file paths and lengths.")

    # Test conditioned model
    print("\nTesting Conditioned Dataset:")
    cond_params = {'param1': 0.75, 'param2': 0.33}
    dataset_cond = AudioDataset(
        input_audio_paths=[dummy_input_path],
        target_audio_paths=[dummy_target_path],
        sequence_length=1024,
        sample_rate=dummy_sr,
        conditioning_params=cond_params
    )
    if len(dataset_cond) > 0:
        dataloader_cond = DataLoader(dataset_cond, batch_size=4, shuffle=True)
        for i, (in_batch, tgt_batch) in enumerate(dataloader_cond):
            print(f"Batch {i+1} (Cond): Input shape: {in_batch.shape}, Target shape: {tgt_batch.shape}")
            # Expect Input shape: [batch_size, sequence_length, 3] (audio + 2 params)
            # Expect Target shape: [batch_size, sequence_length, 1]
            assert in_batch.ndim == 3 and in_batch.shape[2] == 3
            assert tgt_batch.ndim == 3 and tgt_batch.shape[2] == 1
            break
    else:
        print("Conditioned dataset is empty. Check audio file paths and lengths.")

    print("\nDataset tests complete.")
    # Clean up dummy files (optional)
    # os.remove(dummy_input_path)
    # os.remove(dummy_target_path)
    # os.rmdir(temp_data_dir)
