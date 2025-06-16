# new_model_trainer/tests/test_dataset.py
import unittest
import torch
import numpy as np
import os
import soundfile as sf # For creating dummy audio files

# Attempt to import the target module
try:
    from new_core_audio_ml.dataset import AudioDataset
except ImportError:
    AudioDataset = None # Placeholder if import fails in limited env

class TestAudioDataset(unittest.TestCase):

    def setUp(self):
        """Set up dummy audio files for testing."""
        self.test_dir = "temp_test_audio_data"
        os.makedirs(self.test_dir, exist_ok=True)
        self.sample_rate = 48000
        self.seq_len = 1024

        self.input_path1 = os.path.join(self.test_dir, "input1.wav")
        self.target_path1 = os.path.join(self.test_dir, "target1.wav")
        # Create 2 seconds of audio
        sf.write(self.input_path1, np.random.randn(self.sample_rate * 2), self.sample_rate)
        sf.write(self.target_path1, np.random.randn(self.sample_rate * 2), self.sample_rate)

        self.input_path_short = os.path.join(self.test_dir, "input_short.wav")
        self.target_path_short = os.path.join(self.test_dir, "target_short.wav")
        # Create 0.01 seconds of audio (shorter than seq_len)
        sf.write(self.input_path_short, np.random.randn(int(self.sample_rate * 0.01)), self.sample_rate)
        sf.write(self.target_path_short, np.random.randn(int(self.sample_rate * 0.01)), self.sample_rate)


    def tearDown(self):
        """Clean up dummy audio files."""
        for f in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, f))
        os.rmdir(self.test_dir)

    @unittest.skipIf(AudioDataset is None, "AudioDataset module not loaded")
    def test_dataset_creation_no_cond(self):
        dataset = AudioDataset([self.input_path1], [self.target_path1], self.seq_len, self.sample_rate)
        self.assertTrue(len(dataset) > 0, "Dataset should not be empty with valid audio.")
        input_sample, target_sample = dataset[0]
        self.assertEqual(input_sample.shape, (self.seq_len, 1))
        self.assertEqual(target_sample.shape, (self.seq_len, 1))

    @unittest.skipIf(AudioDataset is None, "AudioDataset module not loaded")
    def test_dataset_creation_with_cond(self):
        cond_params = {'param1': 0.5, 'param2': -0.5}
        dataset = AudioDataset([self.input_path1], [self.target_path1], self.seq_len, self.sample_rate, conditioning_params=cond_params)
        self.assertTrue(len(dataset) > 0)
        input_sample, target_sample = dataset[0]
        self.assertEqual(input_sample.shape, (self.seq_len, 3)) # Audio + 2 params
        self.assertEqual(target_sample.shape, (self.seq_len, 1))

        # Check if conditioning parameters are somewhat present in the sample
        # This is a basic check, more specific value checks could be added.
        self.assertNotAlmostEqual(input_sample[0, 1].item(), input_sample[0, 0].item())


    @unittest.skipIf(AudioDataset is None, "AudioDataset module not loaded")
    def test_audio_shorter_than_sequence_length(self):
        dataset = AudioDataset([self.input_path_short], [self.target_path_short], self.seq_len, self.sample_rate)
        self.assertEqual(len(dataset), 0, "Dataset should be empty if audio is too short and not handled by padding.")

    @unittest.skipIf(AudioDataset is None, "AudioDataset module not loaded")
    def test_missing_files(self):
        dataset = AudioDataset(["non_existent_input.wav"], ["non_existent_target.wav"], self.seq_len, self.sample_rate)
        self.assertEqual(len(dataset), 0, "Dataset should be empty if audio files are missing.")

if __name__ == '__main__':
    unittest.main()
