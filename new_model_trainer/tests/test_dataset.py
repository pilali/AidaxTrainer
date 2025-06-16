# new_model_trainer/tests/test_dataset.py
# (Changes shown below, assuming previous setUp and tearDown are kept)
import unittest
import torch
import numpy as np
import os
import soundfile as sf

try:
    from new_core_audio_ml.dataset import AudioDataset
except ImportError:
    AudioDataset = None

class TestAudioDataset(unittest.TestCase):

    def setUp(self):
        self.test_dir = "temp_test_audio_data_cond"
        os.makedirs(self.test_dir, exist_ok=True)
        self.sample_rate = 48000
        self.seq_len = 1024
        self.duration = 2 # seconds

        self.input_path1 = os.path.join(self.test_dir, "input1.wav")
        self.target_path1 = os.path.join(self.test_dir, "target1.wav")
        sf.write(self.input_path1, np.random.randn(self.sample_rate * self.duration), self.sample_rate)
        sf.write(self.target_path1, np.random.randn(self.sample_rate * self.duration), self.sample_rate)

        self.input_path2 = os.path.join(self.test_dir, "input2.wav")
        self.target_path2 = os.path.join(self.test_dir, "target2.wav")
        sf.write(self.input_path2, np.random.randn(self.sample_rate * self.duration), self.sample_rate)
        sf.write(self.target_path2, np.random.randn(self.sample_rate * self.duration), self.sample_rate)

        self.input_path_short = os.path.join(self.test_dir, "input_short.wav")
        sf.write(self.input_path_short, np.random.randn(int(self.sample_rate * 0.01)), self.sample_rate)
        self.target_path_short = os.path.join(self.test_dir, "target_short.wav") # Target for short input
        sf.write(self.target_path_short, np.random.randn(int(self.sample_rate * 0.01)), self.sample_rate)


    def tearDown(self):
        for f_name in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, f_name))
        os.rmdir(self.test_dir)

    @unittest.skipIf(AudioDataset is None, "AudioDataset module not loaded")
    def test_dataset_no_conditioning(self):
        items = [{'input_path': self.input_path1, 'target_path': self.target_path1}]
        dataset = AudioDataset(items, self.seq_len, self.sample_rate, num_conditioning_params=0)
        self.assertTrue(len(dataset) > 0)
        input_sample, target_sample = dataset[0]
        self.assertEqual(input_sample.shape, (self.seq_len, 1)) # Audio only
        self.assertEqual(target_sample.shape, (self.seq_len, 1))

    @unittest.skipIf(AudioDataset is None, "AudioDataset module not loaded")
    def test_dataset_with_conditioning(self):
        items = [
            {'input_path': self.input_path1, 'target_path': self.target_path1, 'conditioning_values': [0.1, 0.9]},
            {'input_path': self.input_path2, 'target_path': self.target_path2, 'conditioning_values': [-0.5, 0.5]}
        ]
        dataset = AudioDataset(items, self.seq_len, self.sample_rate, num_conditioning_params=2)
        self.assertTrue(len(dataset) > 0)

        input_sample, _ = dataset[0] # From first item
        self.assertEqual(input_sample.shape, (self.seq_len, 3)) # Audio + 2 params
        # Check if param values are approximately correct for the first sample of the first item
        self.assertAlmostEqual(input_sample[0, 1].item(), 0.1, places=5)
        self.assertAlmostEqual(input_sample[0, 2].item(), 0.9, places=5)

        # Check a sample from the second item (assuming enough sequences in first item)
        num_seq_item1 = (self.sample_rate * self.duration) // self.seq_len
        if len(dataset) > num_seq_item1 :
            input_sample_item2, _ = dataset[num_seq_item1] # First sample of the second item
            self.assertEqual(input_sample_item2.shape, (self.seq_len, 3))
            self.assertAlmostEqual(input_sample_item2[0, 1].item(), -0.5, places=5)
            self.assertAlmostEqual(input_sample_item2[0, 2].item(), 0.5, places=5)


    @unittest.skipIf(AudioDataset is None, "AudioDataset module not loaded")
    def test_dataset_mismatched_conditioning_params_len(self):
        items = [{'input_path': self.input_path1, 'target_path': self.target_path1, 'conditioning_values': [0.1]}] # Only 1 param value
        # Expecting 2 conditioning params, but only 1 provided for the item
        dataset = AudioDataset(items, self.seq_len, self.sample_rate, num_conditioning_params=2)
        self.assertEqual(len(dataset), 0, "Dataset should be empty if conditioning_values length mismatches num_conditioning_params.")

    @unittest.skipIf(AudioDataset is None, "AudioDataset module not loaded")
    def test_dataset_missing_conditioning_params_when_expected(self):
        items = [{'input_path': self.input_path1, 'target_path': self.target_path1}] # No conditioning_values field
        # Expecting 2 conditioning params
        dataset = AudioDataset(items, self.seq_len, self.sample_rate, num_conditioning_params=2)
        self.assertEqual(len(dataset), 0, "Dataset should be empty if conditioning_values are missing when num_conditioning_params > 0.")

    @unittest.skipIf(AudioDataset is None, "AudioDataset module not loaded")
    def test_audio_shorter_than_sequence_length(self):
        items = [{'input_path': self.input_path_short, 'target_path': self.target_path_short}]
        dataset = AudioDataset(items, self.seq_len, self.sample_rate, num_conditioning_params=0)
        self.assertEqual(len(dataset), 0)

    @unittest.skipIf(AudioDataset is None, "AudioDataset module not loaded")
    def test_missing_files(self):
        items = [{'input_path': "non_existent.wav", 'target_path': "non_existent_target.wav"}]
        dataset = AudioDataset(items, self.seq_len, self.sample_rate, num_conditioning_params=0)
        self.assertEqual(len(dataset), 0)

if __name__ == '__main__':
    unittest.main()
