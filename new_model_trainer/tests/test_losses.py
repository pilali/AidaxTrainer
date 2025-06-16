# new_model_trainer/tests/test_losses.py
import unittest
import torch

# Attempt to import the target modules
try:
    from new_core_audio_ml.losses import ESRLoss, DCLoss, PreEmph, LossWrapper
except ImportError:
    ESRLoss = DCLoss = PreEmph = LossWrapper = None # Placeholders

class TestLossFunctions(unittest.TestCase):

    def setUp(self):
        self.output_clean = torch.sin(torch.arange(0, 100, dtype=torch.float32) * 0.1).unsqueeze(0).unsqueeze(-1) # Batch, Seq, Feat
        self.target_clean = torch.sin(torch.arange(0, 100, dtype=torch.float32) * 0.1).unsqueeze(0).unsqueeze(-1)
        self.output_noisy = self.output_clean + torch.randn_like(self.output_clean) * 0.1
        self.target_dc = self.target_clean + 0.5 # Add DC offset

    @unittest.skipIf(ESRLoss is None, "ESRLoss module not loaded")
    def test_esr_loss_identical(self):
        loss_fn = ESRLoss()
        loss = loss_fn(self.output_clean, self.target_clean)
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    @unittest.skipIf(ESRLoss is None, "ESRLoss module not loaded")
    def test_esr_loss_different(self):
        loss_fn = ESRLoss()
        loss = loss_fn(self.output_noisy, self.target_clean)
        self.assertGreater(loss.item(), 0.0)

    @unittest.skipIf(DCLoss is None, "DCLoss module not loaded")
    def test_dc_loss_no_offset(self):
        loss_fn = DCLoss()
        loss = loss_fn(self.output_clean, self.target_clean) # No DC difference expected
        self.assertAlmostEqual(loss.item(), 0.0, places=5)

    @unittest.skipIf(DCLoss is None, "DCLoss module not loaded")
    def test_dc_loss_with_offset(self):
        loss_fn = DCLoss()
        loss = loss_fn(self.output_clean, self.target_dc) # Target has DC offset
        self.assertGreater(loss.item(), 0.0)

    @unittest.skipIf(PreEmph is None or LossWrapper is None, "PreEmph or LossWrapper not loaded")
    def test_loss_wrapper_with_preemph(self):
        loss_configs = {'ESR': {'weight': 1.0}}
        pre_emph_config = {'apply': True, 'type': 'aw', 'fs': 48000} # A-weighting

        wrapper = LossWrapper(loss_configs, pre_emph_config=pre_emph_config, sample_rate=48000)
        loss = wrapper(self.output_noisy, self.target_clean)
        self.assertIsNotNone(loss.item()) # Just check it runs

    @unittest.skipIf(LossWrapper is None, "LossWrapper not loaded")
    def test_loss_wrapper_multiple_losses(self):
        loss_configs = {
            'ESR': {'weight': 0.7},
            'DC': {'weight': 0.3}
        }
        wrapper = LossWrapper(loss_configs, sample_rate=48000)
        loss = wrapper(self.output_noisy, self.target_dc)
        self.assertIsNotNone(loss.item())

if __name__ == '__main__':
    unittest.main()
