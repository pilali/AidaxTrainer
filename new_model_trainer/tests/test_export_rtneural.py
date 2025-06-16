# new_model_trainer/tests/test_export_rtneural.py
import unittest
import torch
import numpy as np
import json
import os

# Attempt to import target modules
try:
    from new_core_audio_ml.networks import SimpleLSTM
    from new_core_audio_ml.export_rtneural import export_model_to_rtneural_json, _convert_numpy_to_list
except ImportError:
    SimpleLSTM = export_model_to_rtneural_json = _convert_numpy_to_list = None # Placeholders


class TestExportRTNeural(unittest.TestCase):
    def setUp(self):
        self.test_dir = "temp_test_export_data"
        os.makedirs(self.test_dir, exist_ok=True)
        self.sample_rate = 48000

        if SimpleLSTM is not None: # Only create model if class is available
            self.dummy_model_cond = SimpleLSTM(input_size=3, hidden_size=8, num_layers=1, output_size=1, skip_connection=True)
            self.dummy_model_no_cond = SimpleLSTM(input_size=1, hidden_size=8, num_layers=1, output_size=1, skip_connection=False)
            # Put models in eval mode
            self.dummy_model_cond.eval()
            self.dummy_model_no_cond.eval()


    def tearDown(self):
        for f in os.listdir(self.test_dir):
            os.remove(os.path.join(self.test_dir, f))
        os.rmdir(self.test_dir)

    @unittest.skipIf(SimpleLSTM is None or export_model_to_rtneural_json is None, "Required modules not loaded")
    def test_export_structure_conditioned_model(self):
        model = self.dummy_model_cond
        output_path = os.path.join(self.test_dir, "exported_cond_model.aidax")

        export_model_to_rtneural_json(
            pytorch_model=model,
            output_aidax_path=output_path,
            sample_rate=self.sample_rate,
            model_name="TestCondLSTM",
            esr_val=0.1
        )
        self.assertTrue(os.path.exists(output_path))

        with open(output_path, 'r') as f:
            data = json.load(f)

        self.assertEqual(data["in_shape"], [None, None, 3])
        self.assertEqual(data["in_skip"], 1) # skip_connection=True
        self.assertEqual(len(data["layers"]), 2) # 1 LSTM + 1 Dense
        self.assertEqual(data["layers"][0]["type"], "lstm")
        self.assertEqual(data["layers"][0]["shape"], [None, None, model.hidden_size])
        self.assertEqual(data["layers"][1]["type"], "dense")
        self.assertEqual(data["layers"][1]["shape"], [None, None, model.output_size])
        self.assertIn("samplerate", data["metadata"])
        self.assertEqual(data["metadata"]["samplerate"], self.sample_rate)

    @unittest.skipIf(SimpleLSTM is None or export_model_to_rtneural_json is None, "Required modules not loaded")
    def test_export_structure_no_conditioned_model(self):
        model = self.dummy_model_no_cond
        output_path = os.path.join(self.test_dir, "exported_no_cond_model.aidax")

        export_model_to_rtneural_json(
            pytorch_model=model,
            output_aidax_path=output_path,
            sample_rate=self.sample_rate,
            model_name="TestNoCondLSTM"
        )
        self.assertTrue(os.path.exists(output_path))

        with open(output_path, 'r') as f:
            data = json.load(f)

        self.assertEqual(data["in_shape"], [None, None, 1])
        self.assertEqual(data["in_skip"], 0) # skip_connection=False
        self.assertEqual(len(data["layers"]), 2) # 1 LSTM + 1 Dense


    @unittest.skipIf(_convert_numpy_to_list is None, "_convert_numpy_to_list not loaded")
    def test_numpy_to_list_conversion(self):
        data = {
            "a": np.array([1, 2, 3]),
            "b": [np.array([4.0, 5.0]), {"c": np.array([[6, 7], [8, 9]])}]
        }
        converted = _convert_numpy_to_list(data)
        self.assertIsInstance(converted["a"], list)
        self.assertIsInstance(converted["b"][0], list)
        self.assertIsInstance(converted["b"][1]["c"], list)
        self.assertEqual(converted["a"][0], 1)
        self.assertEqual(converted["b"][0][0], 4.0)
        self.assertEqual(converted["b"][1]["c"][0][0], 6)


if __name__ == '__main__':
    unittest.main()
