# New AIDA-X Model Trainer

## Overview

This directory contains a new, modernized Python-based trainer for generating neural network models compatible with the AidaDSP LV2 plugin (AIDAX format). It aims to replace the previous `Automated-GuitarAmpModelling-aidadsp_devel` trainer with updated dependencies and more flexible GPU support.

The trainer takes input (dry) and target (processed) audio files, trains a neural network (currently focused on LSTM-based architectures), and exports the trained model to an `.aidax` JSON file that can be loaded by AidaDSP plugins.

## Features

*   Built with modern PyTorch.
*   Supports training of standard and conditioned neural models.
*   Exports models in AidaDSP LV2 compatible `.aidax` (JSON) format.
*   Highly configurable via command-line arguments and JSON configuration files.
*   Optional GPU acceleration:
    *   NVIDIA CUDA
    *   Apple Silicon (MPS)
    *   AMD ROCm (via appropriate PyTorch build)
    *   Intel GPUs (via appropriate PyTorch build with IPEX)
    *   CPU fallback.
*   Includes tools for loss calculation (ESR, DC loss) and pre-emphasis filtering.
*   Basic logging with TensorBoard.
*   Unit tests for core components.

## Prerequisites

*   Python 3.9 or higher is recommended.
*   Dependencies are listed in `requirements.txt`.
*   A C++ compiler might be needed for some PyTorch components if installing from source, though typically not required when installing pre-built PyTorch binaries.

## Installation/Setup

1.  **Navigate to the Trainer Directory:**
    ```bash
    cd new_model_trainer
    ```

2.  **Create a Python Virtual Environment:**
    (Recommended to avoid conflicts with system-wide packages)
    ```bash
    python -m venv .venv
    ```
    Activate it:
    *   On Windows:
        ```bash
        .venv\Scripts\activate
        ```
    *   On macOS/Linux:
        ```bash
        source .venv/bin/activate
        ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **PyTorch Version for GPU Support:**
    The `requirements.txt` file lists `torch`, `torchvision`, and `torchaudio`. This will typically install a version of PyTorch compatible with your system's CUDA drivers if available, or a CPU-only version. For specific GPU hardware, you might need to install a particular PyTorch build. Refer to the "GPU Support" section below and the [official PyTorch website](https://pytorch.org/get-started/locally/) for detailed instructions.

## Data Preparation

*   **Audio Format:** The trainer expects audio files in `.wav` format.
*   **Input/Target Structure:**
    *   Provide one or more input (dry/DI) `.wav` files.
    *   For each input file, provide a corresponding target (amped/processed) `.wav` file.
    *   These can be specified using the `--input_audio_dir` and `--target_audio_dir` arguments, which can point to directories containing these files or directly to single files. If directories are provided, the script will attempt to pair input and target files (e.g., by matching filenames).
*   **Conditioning Parameters (for conditioned models):**
    *   If training a model that accepts conditioning parameters (e.g., amp gain, pedal tone), these parameters need to be incorporated into the training data.
    *   The `AudioDataset` class and `train_model.py` script accept `--conditioning_param1` and `--conditioning_param2`. When provided, these values will be concatenated with the input audio signal for each training step.
    *   To train a model that responds to a *range* of conditioning parameters, you would typically need to:
        1.  Prepare multiple target audio files, each recorded with different settings of the physical device corresponding to these parameters.
        2.  Run the training script multiple times, or adapt it, to train on these varied examples, potentially associating specific parameter values with specific audio file pairs if the model is to learn a mapping across the parameter space. (The current `train_model.py` uses fixed conditioning parameters for a given run).

## Running the Trainer

The main script for training is `train_model.py`.

*   **Basic Command:**
    ```bash
    python train_model.py \
        --input_audio_dir path/to/your/input_files_or_dir \
        --target_audio_dir path/to/your/target_files_or_dir \
        --output_dir path/to/your_training_output \
        --model_name MyAmpModel \
        --epochs 50 \
        --device cuda
    ```

*   **Using a Configuration File:**
    You can define all training parameters in a JSON configuration file and pass it to the script. See `configs/example_config.json` for a template.
    ```bash
    python train_model.py --config_file configs/example_config.json
    ```
    Command-line arguments will override settings in the configuration file.

*   **Key Command-Line Arguments:**
    *   `--input_audio_dir`: Path to input audio(s).
    *   `--target_audio_dir`: Path to target audio(s).
    *   `--output_dir`: Where to save models, logs, etc.
    *   `--config_file`: Path to a JSON config.
    *   `--model_name`: Name for your model.
    *   `--epochs`: Number of training epochs.
    *   `--batch_size`, `--learning_rate`, `--hidden_size`, etc.
    *   `--device`: `cpu`, `cuda`, `mps`.
    *   `--conditioning_param1`, `--conditioning_param2`: For conditioned models.
    *   Use `python train_model.py --help` for a full list of arguments.

## Output

The training process will generate files in the specified `--output_dir` (e.g., `training_output/MyAmpModel/`):

*   **`checkpoints/MODEL_NAME/`**: Saved PyTorch model checkpoints (`MODEL_NAME_latest.pth`, `MODEL_NAME_best.pth`).
*   **`logs/MODEL_NAME/`**: TensorBoard logs for monitoring training progress. Also contains `training_stats.json`.
*   **`MODEL_NAME.aidax`**: The final exported model in RTNeural JSON format, ready to be used with AidaDSP.

## Configuration File Format (`configs/example_config.json`)

The JSON configuration file allows you to specify most of the command-line arguments in one place. Key fields include:

*   `model_name`, `output_dir`
*   `epochs`, `batch_size`, `learning_rate`, `device`, `early_stopping_patience`
*   `hidden_size`, `num_layers` (LSTM parameters), `skip_connection`
*   `sequence_length`, `sample_rate`
*   `conditioning_params`: `null` or an object like `{"param1": 0.5, "param2": 0.3}`.
*   `loss_functions`: Dictionary specifying loss components (e.g., ESR, DC) and their weights.
*   `pre_emphasis`: Configuration for pre-emphasis filtering.
*   `model_version`: Version string for the exported model metadata.

Refer to `configs/example_config.json` for a detailed example.

## GPU Support

The trainer can leverage GPU acceleration for significantly faster training.

*   **Device Selection:** Use the `--device` flag (or `device` field in config JSON) to specify the computation device:
    *   `--device cuda`: For NVIDIA GPUs.
    *   `--device mps`: For Apple Silicon (M1/M2) GPUs.
    *   `--device cpu`: To run on the CPU.
    If a requested GPU device is not available, the trainer will fall back to CPU.

*   **Installing PyTorch with GPU Support:**
    *   **NVIDIA CUDA:** Ensure you have NVIDIA drivers and CUDA Toolkit installed. Then, install the CUDA-enabled PyTorch build from the [official PyTorch website](https://pytorch.org/get-started/locally/). The command usually looks like:
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cuXXX
        ```
        (Replace `cuXXX` with your CUDA version, e.g., `cu118`, `cu121`).
    *   **Apple Silicon (MPS):** Recent versions of PyTorch (typically 1.12+) include MPS support for Apple Silicon GPUs. Installation is usually standard:
        ```bash
        pip install torch torchvision torchaudio
        ```
        Using `--device mps` should then work.
    *   **AMD ROCm:** For AMD GPUs, you need a PyTorch build with ROCm support. Follow instructions on the PyTorch website. Once installed, you can typically use `--device cuda` (as ROCm often emulates CUDA for PyTorch) or potentially `--device rocm` if your PyTorch build explicitly supports that string.
    *   **Intel GPUs:** Intel provides the [Intel® Extension for PyTorch (IPEX)](https://intel.github.io/intel-extension-for-pytorch/). Follow their installation guide. The device string might be `cuda` (if IPEX maps to it) or a specific string like `xpu`.

## Running Tests

Unit tests are provided to verify the functionality of core components.

1.  **Setup Environment:** Ensure you have activated your virtual environment and installed all dependencies from `requirements.txt`, including a working PyTorch installation.

2.  **Run Tests:** Navigate to the `new_model_trainer` directory and run:
    ```bash
    python -m unittest discover -s tests -p "test_*.py"
    ```
    This will discover and run all test files (named `test_*.py`) in the `tests/` directory.

### Manual End-to-End Test

Since automated end-to-end testing (including model loading in an LV2 host) is complex for this environment, here's a conceptual manual test:

1.  **Prepare Sample Data:** Create a small `input.wav` and a corresponding `target.wav` file (e.g., a few seconds long). Place them in a directory, e.g., `sample_data/input/` and `sample_data/target/`.
2.  **Create a Test Configuration:** Copy `configs/example_config.json` to `configs/test_run_config.json`. Edit it to:
    *   Point to your sample data directories.
    *   Reduce `epochs` to a small number (e.g., 2-3) for a quick run.
    *   Set `batch_size` to a small value (e.g., 2-4).
    *   Set `output_dir` to a temporary location like `temp_test_output`.
3.  **Run Trainer:**
    ```bash
    python train_model.py --config_file configs/test_run_config.json
    ```
4.  **Check Output:**
    *   Verify that the `temp_test_output` directory is created.
    *   Look for model checkpoints (`.pth` files) and TensorBoard logs.
    *   Crucially, check for the exported `.aidax` file.
5.  **Inspect `.aidax` File:** Open the `.aidax` file with a text editor. Verify its basic JSON structure and that it contains model weights and metadata.
6.  **(External) Test in AidaDSP:** If possible, load the generated `.aidax` file into the AidaDSP LV2 plugin in a compatible host to ensure it loads and processes audio.

This completes the primary documentation for the new model trainer.

## Colab Notebook for Training

A Jupyter Notebook named `New_AIDA_X_Trainer_Colab.ipynb` is available within this (`new_model_trainer`) directory. This notebook provides an interactive environment to:
*   Set up the necessary dependencies.
*   Connect to your Google Drive for data and configuration.
*   Configure and run the training script.
*   Download the final `.aidax` model.

This is particularly useful for leveraging Google Colab's free GPU resources for training. To use it, upload this notebook to your Google Colab account, or navigate to it if you have cloned the repository in Colab.
