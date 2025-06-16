import torch
import torch.nn as nn
import json
import os

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, output_size=1, skip_connection=True):
        """
        A simple LSTM model with a final Linear layer.

        Args:
            input_size (int): Number of input features.
                              (1 for non-conditioned, 3 for audio + 2 params).
            hidden_size (int): The number of features in the hidden state h.
            num_layers (int): Number of recurrent layers.
            output_size (int): Number of output features (typically 1 for audio).
            skip_connection (bool): If True, add a skip connection from input to output.
        """
        super(SimpleLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.skip_connection = skip_connection

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True) # batch_first=True expects input: (batch, seq, feature)

        self.linear = nn.Linear(hidden_size, output_size)

        self.hidden_cell = None # For storing (h_n, c_n)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, output_size).
        """
        # LSTM out: (batch, seq_len, hidden_size)
        # self.hidden_cell is (h_n, c_n) where h_n and c_n are (num_layers, batch, hidden_size)
        lstm_out, self.hidden_cell = self.lstm(x, self.hidden_cell)

        # Pass LSTM output through the linear layer
        # Linear layer expects input (N, *, H_in) where H_in is hidden_size
        # Output is (N, *, H_out) where H_out is output_size
        predictions = self.linear(lstm_out)

        if self.skip_connection:
            # Add the first feature of the input (assumed to be audio) to the predictions
            # Input x is (batch, seq_len, input_size)
            # We want to add x[:, :, 0] which is (batch, seq_len)
            # Predictions is (batch, seq_len, output_size) which is typically (batch, seq_len, 1)
            # Ensure dimensions match for broadcasting or direct addition
            if self.input_size >= 1: # Ensure there's at least one input feature for skip
                 # Unsqueeze to make it (batch, seq_len, 1) to match predictions shape
                skip_input = x[:, :, 0].unsqueeze(-1)
                predictions = predictions + skip_input
            else:
                # This case should ideally not happen if skip_connection is True with input_size < 1
                pass


        return predictions

    def detach_hidden(self):
        """Detaches the hidden state from the computation graph."""
        if self.hidden_cell is not None:
            self.hidden_cell = (self.hidden_cell[0].detach(), self.hidden_cell[1].detach())

    def reset_hidden(self, batch_size=None, device=None):
        """Resets the hidden state to None or zeros."""
        # If batch_size and device are provided, initialize to zeros,
        # otherwise set to None to let LSTM layer initialize it on first forward pass.
        if batch_size and device:
            h_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            c_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            self.hidden_cell = (h_0, c_0)
        else:
            self.hidden_cell = None

    def save_pytorch_model(self, file_path):
        """Saves the model's state_dict and architecture config to a file."""
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

        model_config = {
            'model_name': self.__class__.__name__,
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'output_size': self.output_size,
            'skip_connection': self.skip_connection
        }

        save_content = {
            'model_config': model_config,
            'state_dict': self.state_dict()
        }
        torch.save(save_content, file_path)
        print(f"PyTorch model saved to {file_path}")

    @classmethod
    def load_pytorch_model(cls, file_path):
        """Loads a model from a file containing state_dict and architecture config."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")

        checkpoint = torch.load(file_path)
        model_config = checkpoint['model_config']

        # Ensure the loaded config is for this model type
        if model_config.get('model_name') != cls.__name__:
            raise ValueError(f"File {file_path} contains model type {model_config.get('model_name')}, expected {cls.__name__}")

        model = cls(input_size=model_config['input_size'],
                    hidden_size=model_config['hidden_size'],
                    num_layers=model_config['num_layers'],
                    output_size=model_config['output_size'],
                    skip_connection=model_config['skip_connection'])

        model.load_state_dict(checkpoint['state_dict'])
        print(f"PyTorch model loaded from {file_path}")
        return model

# Example Usage (for testing purposes)
if __name__ == '__main__':
    # Test parameters
    batch_s = 4
    seq_len = 100

    # Non-conditioned model test
    print("Testing Non-Conditioned SimpleLSTM:")
    input_s_no_cond = 1
    model_no_cond = SimpleLSTM(input_size=input_s_no_cond, hidden_size=16, num_layers=1, output_size=1, skip_connection=True)
    dummy_input_no_cond = torch.randn(batch_s, seq_len, input_s_no_cond) # (batch, seq, feature)
    model_no_cond.reset_hidden(batch_s, dummy_input_no_cond.device) # Initialize hidden state
    output_no_cond = model_no_cond(dummy_input_no_cond)
    print(f"Input shape (No Cond): {dummy_input_no_cond.shape}")
    print(f"Output shape (No Cond): {output_no_cond.shape}")
    # Expected output: (batch_s, seq_len, 1)
    assert output_no_cond.shape == (batch_s, seq_len, 1)

    # Conditioned model test
    print("\nTesting Conditioned SimpleLSTM:")
    input_s_cond = 3 # Audio + 2 parameters
    model_cond = SimpleLSTM(input_size=input_s_cond, hidden_size=16, num_layers=1, output_size=1, skip_connection=True)
    dummy_input_cond = torch.randn(batch_s, seq_len, input_s_cond) # (batch, seq, feature)
    model_cond.reset_hidden(batch_s, dummy_input_cond.device) # Initialize hidden state
    output_cond = model_cond(dummy_input_cond)
    print(f"Input shape (Cond): {dummy_input_cond.shape}")
    print(f"Output shape (Cond): {output_cond.shape}")
    # Expected output: (batch_s, seq_len, 1)
    assert output_cond.shape == (batch_s, seq_len, 1)

    print("\nTesting Skip Connection (No Cond, skip=False):")
    model_no_skip = SimpleLSTM(input_size=input_s_no_cond, hidden_size=16, num_layers=1, output_size=1, skip_connection=False)
    model_no_skip.reset_hidden(batch_s, dummy_input_no_cond.device)
    output_no_skip = model_no_skip(dummy_input_no_cond)
    print(f"Input shape (No Skip): {dummy_input_no_cond.shape}")
    print(f"Output shape (No Skip): {output_no_skip.shape}")
    assert output_no_skip.shape == (batch_s, seq_len, 1)

    # Test model saving and loading
    print("\nTesting Model Save/Load:")
    temp_model_dir = "temp_model_checkpoints"
    if not os.path.exists(temp_model_dir):
        os.makedirs(temp_model_dir)

    model_path = os.path.join(temp_model_dir, "test_simple_lstm.pth")
    model_cond.save_pytorch_model(model_path)
    loaded_model = SimpleLSTM.load_pytorch_model(model_path)

    # Verify loaded model config
    assert loaded_model.input_size == model_cond.input_size
    assert loaded_model.hidden_size == model_cond.hidden_size
    assert loaded_model.skip_connection == model_cond.skip_connection
    print("Model saving and loading successful.")

    # Clean up
    # os.remove(model_path)
    # os.rmdir(temp_model_dir)

    print("\nNetwork tests complete.")
