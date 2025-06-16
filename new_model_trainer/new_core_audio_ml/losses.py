import torch
import torch.nn as nn
from auraloss.perceptual import FIRFilter # Using auraloss for pre-emphasis

class ESRLoss(nn.Module):
    """
    Error-to-Signal Ratio Loss.
    Calculates mean((target - output)^2) / (mean(target^2) + epsilon).
    """
    def __init__(self, epsilon=1e-8):
        super(ESRLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, output, target):
        # Ensure target and output are of the same shape
        if output.shape != target.shape:
            raise ValueError(f"Output shape {output.shape} must match target shape {target.shape}")

        loss_mse = torch.mean(torch.pow(target - output, 2))
        energy_target = torch.mean(torch.pow(target, 2))

        loss = loss_mse / (energy_target + self.epsilon)
        return loss

class DCLoss(nn.Module):
    """
    DC Loss.
    Calculates mean((mean(target) - mean(output))^2) / (mean(target^2) + epsilon).
    This penalizes differences in the DC offset.
    """
    def __init__(self, epsilon=1e-8):
        super(DCLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, output, target):
        if output.shape != target.shape:
            raise ValueError(f"Output shape {output.shape} must match target shape {target.shape}")

        dc_loss_mse = torch.pow(torch.mean(target) - torch.mean(output), 2)
        # Alternative: calculate mean DC diff per batch item if inputs are (batch, seq, feat)
        # dc_loss_mse = torch.mean(torch.pow(torch.mean(target, dim=1) - torch.mean(output, dim=1), 2))

        energy_target = torch.mean(torch.pow(target, 2))

        loss = dc_loss_mse / (energy_target + self.epsilon)
        return loss

class PreEmph(nn.Module):
    """
    Pre-emphasis filter module using auraloss.FIRFilter.
    Applies filtering to both output and target signals.
    Expects input shape (batch, channels, samples) for FIRFilter.
    """
    def __init__(self, filter_type='hp', fs=48000, **kwargs):
        super(PreEmph, self).__init__()
        # Example: for 'hp', coef might be (5.9659e+03 * 2 * torch.pi) / fs
        # For 'aw' (A-weighting), coef is not needed by auraloss.
        # kwargs can pass 'coef' or other FIRFilter params.
        self.preemph_filter = FIRFilter(filter_type=filter_type, fs=fs, **kwargs)

    def forward(self, output, target):
        # Assuming input tensors are (batch_size, sequence_length, num_features=1)
        # Permute to (batch_size, num_features, sequence_length) for FIRFilter
        if output.ndim != 3 or target.ndim != 3:
            raise ValueError("Inputs to PreEmph must be 3D tensors (batch, seq, feat)")

        output_permuted = output.permute(0, 2, 1)
        target_permuted = target.permute(0, 2, 1)

        filtered_output, filtered_target = self.preemph_filter(output_permuted, target_permuted)

        # Permute back to (batch_size, sequence_length, num_features)
        return filtered_output.permute(0, 2, 1), filtered_target.permute(0, 2, 1)

class LossWrapper(nn.Module):
    """
    Wraps multiple loss functions and combines them with specified weights.
    Handles optional pre-emphasis filtering.
    """
    def __init__(self, loss_configs, pre_emph_config=None, sample_rate=48000):
        super(LossWrapper, self).__init__()
        self.loss_functions = nn.ModuleList()
        self.loss_weights = []
        self.pre_emph = None

        if pre_emph_config and pre_emph_config.get('apply', False):
            self.pre_emph = PreEmph(
                filter_type=pre_emph_config.get('type', 'hp'),
                fs=sample_rate,
                coef=pre_emph_config.get('coef') # Pass coef only if it exists
            )

        for name, config in loss_configs.items():
            weight = config.get('weight', 1.0)
            params = config.get('params', {})

            if name.upper() == 'ESR':
                self.loss_functions.append(ESRLoss(**params))
                self.loss_weights.append(weight)
            elif name.upper() == 'DC':
                self.loss_functions.append(DCLoss(**params))
                self.loss_weights.append(weight)
            # Add other losses here if needed
            else:
                print(f"Warning: Unknown loss function '{name}' in config. Skipping.")

        if not self.loss_functions:
            raise ValueError("No valid loss functions configured.")

        self.loss_weights = torch.tensor(self.loss_weights)

    def forward(self, output, target):
        if self.pre_emph:
            processed_output, processed_target = self.pre_emph(output, target)
        else:
            processed_output, processed_target = output, target

        total_loss = 0.0
        for i, loss_fn in enumerate(self.loss_functions):
            total_loss += self.loss_weights[i].to(output.device) * loss_fn(processed_output, processed_target)

        return total_loss

# Example Usage (for testing purposes)
if __name__ == '__main__':
    batch_size = 4
    sequence_length = 1024
    num_features = 1
    sample_rate_test = 48000

    # Create dummy output and target tensors
    # Shape: (batch_size, sequence_length, num_features)
    dummy_output = torch.randn(batch_size, sequence_length, num_features) * 0.5
    dummy_target = torch.randn(batch_size, sequence_length, num_features) * 0.5 + 0.1 # Add some DC offset and diff

    print("Testing Individual Losses:")
    esr_loss_fn = ESRLoss()
    dc_loss_fn = DCLoss()

    loss_esr = esr_loss_fn(dummy_output, dummy_target)
    loss_dc = dc_loss_fn(dummy_output, dummy_target)
    print(f"ESRLoss: {loss_esr.item()}")
    print(f"DCLoss: {loss_dc.item()}")

    print("\nTesting PreEmph:")
    # A-weighting doesn't require a coefficient typically from user
    pre_emph_aw = PreEmph(filter_type='aw', fs=sample_rate_test)
    out_emph_aw, tgt_emph_aw = pre_emph_aw(dummy_output, dummy_target)
    print(f"Shape after A-weighting PreEmph: Output {out_emph_aw.shape}, Target {tgt_emph_aw.shape}")
    assert out_emph_aw.shape == dummy_output.shape

    # HP filter might need a coefficient (example for auraloss FIRFilter)
    # This coefficient calculation is illustrative; actual values depend on desired cutoff
    hp_coef_example = (1000.0 * 2 * torch.pi) / sample_rate_test
    pre_emph_hp = PreEmph(filter_type='hp', fs=sample_rate_test, coef=hp_coef_example)
    out_emph_hp, tgt_emph_hp = pre_emph_hp(dummy_output, dummy_target)
    print(f"Shape after HP PreEmph: Output {out_emph_hp.shape}, Target {tgt_emph_hp.shape}")
    assert out_emph_hp.shape == dummy_output.shape


    print("\nTesting LossWrapper:")

    # Configuration for LossWrapper
    loss_configurations = {
        'ESR': {'weight': 0.75, 'params': {'epsilon': 1e-7}},
        'DC': {'weight': 0.25}
    }

    # No pre-emphasis
    print("LossWrapper (No PreEmph):")
    loss_wrapper_no_emph = LossWrapper(loss_configs=loss_configurations, sample_rate=sample_rate_test)
    total_loss_no_emph = loss_wrapper_no_emph(dummy_output, dummy_target)
    print(f"Total Loss (No PreEmph): {total_loss_no_emph.item()}")

    # With A-weighting pre-emphasis
    print("LossWrapper (A-Weighting PreEmph):")
    pre_emphasis_config_aw = {'apply': True, 'type': 'aw'}
    loss_wrapper_aw = LossWrapper(loss_configs=loss_configurations,
                                pre_emph_config=pre_emphasis_config_aw,
                                sample_rate=sample_rate_test)
    total_loss_aw = loss_wrapper_aw(dummy_output, dummy_target)
    print(f"Total Loss (A-Weighting PreEmph): {total_loss_aw.item()}")

    # With HP pre-emphasis
    print("LossWrapper (HP PreEmph):")
    pre_emphasis_config_hp = {'apply': True, 'type': 'hp', 'coef': hp_coef_example}
    loss_wrapper_hp = LossWrapper(loss_configs=loss_configurations,
                                pre_emph_config=pre_emphasis_config_hp,
                                sample_rate=sample_rate_test)
    total_loss_hp = loss_wrapper_hp(dummy_output, dummy_target)
    print(f"Total Loss (HP PreEmph): {total_loss_hp.item()}")

    print("\nLosses tests complete.")
