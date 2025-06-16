# new_model_trainer/tests/test_networks.py
import unittest
import torch
import os

# Attempt to import the target module
try:
    from new_core_audio_ml.networks import SimpleLSTM
except ImportError:
    SimpleLSTM = None # Placeholder

class TestSimpleLSTM(unittest.TestCase):

    @unittest.skipIf(SimpleLSTM is None, "SimpleLSTM module not loaded")
    def test_network_creation_and_forward_no_cond(self):
        model = SimpleLSTM(input_size=1, hidden_size=16, num_layers=1, output_size=1, skip_connection=True)
        model.eval()
        dummy_input = torch.randn(2, 100, 1) # Batch, Seq, Features
        model.reset_hidden(batch_size=2, device=dummy_input.device)
        output = model(dummy_input)
        self.assertEqual(output.shape, (2, 100, 1))

    @unittest.skipIf(SimpleLSTM is None, "SimpleLSTM module not loaded")
    def test_network_creation_and_forward_with_cond(self):
        model = SimpleLSTM(input_size=3, hidden_size=16, num_layers=1, output_size=1, skip_connection=True)
        model.eval()
        dummy_input = torch.randn(2, 100, 3) # Batch, Seq, Features (Audio + 2 params)
        model.reset_hidden(batch_size=2, device=dummy_input.device)
        output = model(dummy_input)
        self.assertEqual(output.shape, (2, 100, 1))

    @unittest.skipIf(SimpleLSTM is None, "SimpleLSTM module not loaded")
    def test_network_forward_no_skip(self):
        model = SimpleLSTM(input_size=1, hidden_size=16, num_layers=1, output_size=1, skip_connection=False)
        model.eval()
        dummy_input = torch.randn(2, 100, 1)
        model.reset_hidden(batch_size=2, device=dummy_input.device)
        output = model(dummy_input)
        self.assertEqual(output.shape, (2, 100, 1))
        # Further tests could check if output is different from skip=True version

    @unittest.skipIf(SimpleLSTM is None, "SimpleLSTM module not loaded")
    def test_model_save_load(self):
        model = SimpleLSTM(input_size=3, hidden_size=8, num_layers=2, output_size=1, skip_connection=False)
        test_model_dir = "temp_test_model_save"
        os.makedirs(test_model_dir, exist_ok=True)
        model_path = os.path.join(test_model_dir, "test_lstm_model.pth")

        model.save_pytorch_model(model_path)
        self.assertTrue(os.path.exists(model_path))

        loaded_model = SimpleLSTM.load_pytorch_model(model_path)
        self.assertIsNotNone(loaded_model)
        self.assertEqual(loaded_model.input_size, 3)
        self.assertEqual(loaded_model.hidden_size, 8)
        self.assertEqual(loaded_model.lstm.num_layers, 2)
        self.assertFalse(loaded_model.skip_connection)

        # Clean up
        os.remove(model_path)
        os.rmdir(test_model_dir)

if __name__ == '__main__':
    unittest.main()
