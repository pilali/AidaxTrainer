import json
import torch
import torch.nn as nn # For nn.LSTM and nn.Linear to compare names if needed
import numpy as np
import os

# Attempt to import SimpleLSTM, handle if not found during subtask execution
try:
    from .networks import SimpleLSTM
except ImportError:
    print("Warning: Could not import SimpleLSTM from .networks. Using placeholder if in subtask.")
    # Define a placeholder if SimpleLSTM cannot be imported (e.g., during subtask)
    class SimpleLSTM(nn.Module): # Minimal placeholder
        def __init__(self, input_size, hidden_size, num_layers=1, output_size=1, skip_connection=True):
            super().__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.num_layers = num_layers # Store for RTNeural export
            self.output_size = output_size
            self.skip_connection = skip_connection
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.linear = nn.Linear(hidden_size, output_size)
        def forward(self, x): return self.linear(self.lstm(x)[0])


def _convert_numpy_to_list(obj):
    """Recursively convert numpy arrays to lists in a given object."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [_convert_numpy_to_list(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: _convert_numpy_to_list(value) for key, value in obj.items()}
    else:
        return obj

def export_model_to_rtneural_json(
    pytorch_model,
    output_aidax_path,
    sample_rate,
    model_name="UnnamedModel",
    model_version="0.1.0",
    esr_val=None,
    input_batch_example=None, # Example: list of floats
    output_batch_example=None # Example: list of floats
    ):
    """
    Exports a trained SimpleLSTM PyTorch model to the RTNeural JSON (.aidax) format.

    Args:
        pytorch_model (SimpleLSTM): The trained PyTorch model instance.
        output_aidax_path (str): Path to save the output .aidax JSON file.
        sample_rate (int): Sample rate of the model.
        model_name (str): Name of the model.
        model_version (str): Version of the model.
        esr_val (float, optional): Achieved ESR on validation set.
        input_batch_example (list, optional): An example input batch (list of floats).
        output_batch_example (list, optional): An example output batch (list of floats).
    """
    if not isinstance(pytorch_model, SimpleLSTM):
        raise ValueError("This export function currently only supports SimpleLSTM models.")

    state_dict = pytorch_model.state_dict()

    rtneural_model_dict = {
        "in_shape": [None, None, pytorch_model.input_size],
        "in_skip": 1 if pytorch_model.skip_connection else 0, # RTNeural expects int
        "layers": []
    }

    # LSTM Layer Export (assuming model has exactly one LSTM layer as defined in SimpleLSTM)
    # PyTorch LSTM weights are named like:
    # lstm.weight_ih_l0 (input-hidden weights for layer 0)
    # lstm.weight_hh_l0 (hidden-hidden weights for layer 0)
    # lstm.bias_ih_l0   (input-hidden biases for layer 0)
    # lstm.bias_hh_l0   (hidden-hidden biases for layer 0)
    # For multi-layer LSTMs, the _l{N} suffix changes. We assume num_layers=1 for now for direct mapping.
    # If pytorch_model.num_layers > 1, this part needs careful adaptation for RTNeural's multi-layer format.
    # The original modelToRTNeural.py iterates num_layers for "rec.weight_ih_l%d", "rec.bias_ih_l%d" etc.
    # Our SimpleLSTM has self.lstm which is a single nn.LSTM module.
    # If self.lstm.num_layers > 1, PyTorch handles it internally. We need to extract per-layer weights if RTNeural expects them separately.
    # For now, let's assume RTNeural expects weights for each layer if num_layers > 1 in a similar loop.

    # For simplicity, the current SimpleLSTM class uses a single nn.LSTM module.
    # If num_layers > 1 for this nn.LSTM, PyTorch's state_dict stores them with _l0, _l1 etc.
    # RTNeural's format from the old script implies separate layer definitions in the JSON.
    # This needs reconciliation. For now, this export will assume num_layers in SimpleLSTM means
    # we should iterate and expect weights for each layer like rec.weight_ih_l{idx}.
    # However, our SimpleLSTM uses a single nn.LSTM(num_layers=pytorch_model.num_layers)
    # The weights in its state_dict are for the *entire* multi-layer block if num_layers > 1.
    # This is a key difference from the old format if RTNeural expects discrete layer JSON objects.
    # For now, let's assume RTNeural can take the PyTorch multi-layer LSTM weights directly for a *single* LSTM JSON object.
    # If not, SimpleLSTM needs to be a ModuleList of single-layer LSTMs.

    # Based on original modelToRTNeural.py, it iterates through num_layers
    # and constructs one LSTM JSON layer object per PyTorch layer.
    # This implies the PyTorch model should also be structured as a sequence of single-layer LSTMs
    # if we are to match that export format directly.
    # For now, adhering to the SimpleLSTM having one nn.LSTM module which can be multi-layer:
    # And assuming RTNeural's LSTM layer can parse PyTorch's multi-layer format for a single nn.LSTM block.
    # This is the most straightforward interpretation of the current SimpleLSTM.

    # Re-evaluating: The old script's loop `for num in range(0, num_layers):` and
    # `model_data['state_dict']['rec.weight_ih_l%d' % num]` strongly suggests the old
    # PyTorch model had a structure where 'rec' was a ModuleList or similar, and each 'l%d' was a separate layer.
    # Our current `SimpleLSTM.lstm` is one `nn.LSTM` module.
    # If `pytorch_model.num_layers == 1` for `self.lstm`, then the keys are `lstm.weight_ih_l0`, etc.

    # For now, let's assume pytorch_model.num_layers in SimpleLSTM refers to the number of layers *within* the single nn.LSTM module.
    # And that RTNeural's format expects one JSON "lstm" layer object, and can handle PyTorch's multi-layer weights if provided.
    # If RTNeural expects one JSON object per actual layer, this export needs to be more complex or SimpleLSTM needs redesign.

    # Simplification: Assume pytorch_model.num_layers is 1 for the first pass to match RTNeural's single LSTM layer object.
    # This means the SimpleLSTM should be configured with num_layers=1 for this export to be simple.
    # If SimpleLSTM has num_layers > 1, this export will currently only use l0 weights.

    # Correct approach: The old script creates one JSON layer per layer in the PyTorch model.
    # If our SimpleLSTM uses num_layers > 1, its state_dict has weights like 'lstm.weight_ih_l0', 'lstm.weight_ih_l1', etc.
    # We should iterate `pytorch_model.lstm.num_layers` and create one JSON LSTM object for each.

    for i in range(pytorch_model.lstm.num_layers):
        w_ih = state_dict[f'lstm.weight_ih_l{i}'].cpu().numpy()
        w_hh = state_dict[f'lstm.weight_hh_l{i}'].cpu().numpy()
        b_ih = state_dict[f'lstm.bias_ih_l{i}'].cpu().numpy()
        b_hh = state_dict[f'lstm.bias_hh_l{i}'].cpu().numpy()

        # Weights for RTNeural LSTM: [kernel, recurrent_kernel, bias]
        # kernel = W_i, W_f, W_c, W_o (concatenated input weights)
        # recurrent_kernel = U_i, U_f, U_c, U_o (concatenated recurrent weights)
        # bias = B_i, B_f, B_c, B_o (concatenated input and recurrent biases)
        # PyTorch stores them as (4*hidden_size, input_size) for W, and (4*hidden_size, hidden_size) for U.
        # Bias is (4*hidden_size).
        # RTNeural expects kernel to be (input_size, 4*hidden_size) after transpose.
        # RTNeural expects recurrent_kernel to be (hidden_size, 4*hidden_size) after transpose.
        # RTNeural expects bias to be (4*hidden_size).
        # Original script does: [np.transpose(WVals), np.transpose(UVals), bias_ih_l0 + bias_hh_l0]

        # WVals in old script = rec.weight_ih_l%d -> shape (4*hidden, input_size for layer 0, or 4*hidden, hidden_size for >0)
        # UVals in old script = rec.weight_hh_l%d -> shape (4*hidden, hidden_size)

        # For the first LSTM layer (i=0), input_size is pytorch_model.input_size
        # For subsequent LSTM layers (i > 0), input_size is pytorch_model.hidden_size
        current_lstm_input_size = pytorch_model.input_size if i == 0 else pytorch_model.hidden_size

        # Check shapes from PyTorch
        # w_ih: (4 * hidden_size, current_lstm_input_size)
        # w_hh: (4 * hidden_size, hidden_size)
        # b_ih, b_hh: (4 * hidden_size)

        lstm_layer_rt = {
            "type": "lstm",
            "activation": "",
            "shape": [None, None, pytorch_model.hidden_size], # Output shape of this LSTM layer
            "weights": [
                np.transpose(w_ih), # Kernel: (current_lstm_input_size, 4 * hidden_size)
                np.transpose(w_hh), # Recurrent Kernel: (hidden_size, 4 * hidden_size)
                b_ih + b_hh         # Bias: (4 * hidden_size)
            ]
        }
        rtneural_model_dict["layers"].append(lstm_layer_rt)

    # Dense Layer Export
    lin_weights = state_dict['linear.weight'].cpu().numpy() # Shape (output_size, hidden_size)
    lin_bias = state_dict['linear.bias'].cpu().numpy()     # Shape (output_size)

    # RTNeural dense layer expects weights: [kernel, bias]
    # kernel: (hidden_size, output_size)
    # bias: (output_size)
    # So, PyTorch's linear.weight needs to be transposed.
    dense_layer_rt = {
        "type": "dense",
        "activation": "",
        "shape": [None, None, pytorch_model.output_size],
        "weights": [
            np.transpose(lin_weights),
            lin_bias
        ]
    }
    rtneural_model_dict["layers"].append(dense_layer_rt)

    # Add metadata
    rtneural_model_dict["metadata"] = {
        "samplerate": sample_rate,
        "name": model_name,
        "version": model_version,
        "architecture": pytorch_model.__class__.__name__, # e.g. "SimpleLSTM"
        "trained_by": "new_model_trainer",
        "input_size": pytorch_model.input_size,
        "hidden_size": pytorch_model.hidden_size,
        "num_layers_lstm": pytorch_model.lstm.num_layers, # Number of layers in the nn.LSTM module
        "output_size": pytorch_model.output_size,
        "skip_connection": pytorch_model.skip_connection
    }
    if esr_val is not None:
        rtneural_model_dict["metadata"]["esr_val"] = esr_val
    if input_batch_example is not None:
        rtneural_model_dict["input_batch"] = input_batch_example
    if output_batch_example is not None:
        rtneural_model_dict["output_batch"] = output_batch_example

    # Convert all numpy arrays to lists for JSON serialization
    rtneural_model_dict = _convert_numpy_to_list(rtneural_model_dict)

    # Save to file
    output_dir = os.path.dirname(output_aidax_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_aidax_path, 'w') as f:
        json.dump(rtneural_model_dict, f, indent=4)

    print(f"Model successfully exported to RTNeural format: {output_aidax_path}")


# Example Usage
if __name__ == '__main__':
    print("RTNeural Export Example:")

    # Create a dummy SimpleLSTM model instance
    # Parameters for the dummy model
    example_input_size = 3  # Conditioned model: audio + 2 params
    example_hidden_size = 16
    example_num_lstm_layers = 1 # Number of layers within the nn.LSTM module
    example_output_size = 1
    example_skip_connection = True

    # Instantiate the model
    # Note: If SimpleLSTM is None due to import error in subtask, this will fail.
    if SimpleLSTM is None:
        print("Cannot run export example: SimpleLSTM class not available.")
    else:
        dummy_model = SimpleLSTM(
            input_size=example_input_size,
            hidden_size=example_hidden_size,
            num_layers=example_num_lstm_layers,
            output_size=example_output_size,
            skip_connection=example_skip_connection
        )
        # Models are initialized with random weights by PyTorch by default.
        # For a real scenario, you'd load trained weights:
        # dummy_model.load_state_dict(torch.load('path_to_trained_weights.pth'))
        dummy_model.eval() # Set to evaluation mode

        # Define output path and metadata
        temp_export_dir = "temp_export"
        if not os.path.exists(temp_export_dir):
            os.makedirs(temp_export_dir)
        output_path = os.path.join(temp_export_dir, "dummy_model_rtneural.aidax")

        example_sr = 48000
        example_name = "MyDummyLSTM"
        example_esr = 0.05
        # Dummy batch examples (replace with actual data if available)
        # For input_size=3, sequence_length=5
        example_input_batch = [[ [0.1, 0.5, 0.2], [0.2, 0.5, 0.2], [0.3, 0.5, 0.2], [0.4, 0.5, 0.2], [0.5, 0.5, 0.2] ]]
        example_output_batch = [[ [0.15], [0.25], [0.35], [0.45], [0.55] ]]


        print(f"Exporting dummy model to {output_path}...")
        export_model_to_rtneural_json(
            pytorch_model=dummy_model,
            output_aidax_path=output_path,
            sample_rate=example_sr,
            model_name=example_name,
            esr_val=example_esr,
            input_batch_example=example_input_batch,
            output_batch_example=output_batch_example
        )

        # Verify by loading and checking some keys (optional)
        try:
            with open(output_path, 'r') as f:
                loaded_json = json.load(f)
            print(f"Successfully loaded back exported JSON from {output_path}.")
            print(f"Model type in JSON: {loaded_json['layers'][0]['type']}")
            assert loaded_json['in_shape'][2] == example_input_size
            assert len(loaded_json['layers']) == example_num_lstm_layers + 1 # N LSTM layers + 1 Dense
            assert loaded_json['metadata']['name'] == example_name
            print("Basic verification of exported JSON successful.")
        except Exception as e:
            print(f"Error during verification of exported JSON: {e}")

        # Clean up (optional)
        # os.remove(output_path)
        # os.rmdir(temp_export_dir)
        print("\nExport example complete.")
