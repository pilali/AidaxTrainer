# Raw Content for New_AIDA_X_Trainer_Colab.ipynb

This file contains the raw content for each cell of the Jupyter Notebook.
You can manually create a new notebook in Google Colab and copy/paste the content into respective cells (markdown or code).

---
## Cell 1: Markdown
---

# New AIDA-X Model Trainer for Colab
This notebook will guide you through training a new model using the `new_model_trainer`.
It assumes your new_model_trainer code (including `train_model.py`, `requirements.txt` and the `new_core_audio_ml` library) is accessible, typically by cloning the repository it resides in.

---
## Cell 2: Code - Section 1: Setup Environment
---

```python
#@markdown ### (RUN CELL) 1.1: Initial Setup, Dependencies, and Google Drive
#@markdown This cell will:
#@markdown 1. Clone your repository (if not already in the correct environment).
#@markdown 2. Change directory to `new_model_trainer`.
#@markdown 3. Install required Python packages from `requirements.txt`.
#@markdown 4. Check for GPU and mount Google Drive.

import os
import subprocess
import sys

# --- Repository Setup ---
# IMPORTANT: Replace with your actual repository details if different
GIT_CLONE_URL = "https://github.com/AidaDSP/Automated-GuitarAmpModelling.git" # Or your fork/repo
REPO_NAME = GIT_CLONE_URL.split('/')[-1].replace(".git", "")
COLAB_CONTENT_PATH = "/content"
REPO_PATH = os.path.join(COLAB_CONTENT_PATH, REPO_NAME)
NEW_TRAINER_DIR_NAME = "new_model_trainer" # The directory containing your new trainer
FULL_TRAINER_PATH = os.path.join(REPO_PATH, NEW_TRAINER_DIR_NAME)

print(f"Targeting repository: {REPO_NAME}")
print(f"Targeting trainer directory within repo: {NEW_TRAINER_DIR_NAME}")

if os.path.isdir(FULL_TRAINER_PATH):
    print(f"Trainer directory already exists at: {FULL_TRAINER_PATH}")
    os.chdir(FULL_TRAINER_PATH)
    print(f"Changed directory to: {os.getcwd()}")
    print("Attempting to pull latest changes...")
    try:
        result = subprocess.run(["git", "pull"], check=True, capture_output=True, text=True)
        print("Git pull successful.")
        if result.stdout: print(f"Stdout: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Git pull failed: {e.stderr}")
elif os.path.isdir(REPO_PATH) and not os.path.isdir(FULL_TRAINER_PATH):
    print(f"Repository {REPO_NAME} found, but trainer directory {NEW_TRAINER_DIR_NAME} is missing.")
    print(f"Please ensure '{NEW_TRAINER_DIR_NAME}' exists in your repository's root.")
    sys.exit(f"Trainer directory {NEW_TRAINER_DIR_NAME} not found in {REPO_PATH}")
else:
    print(f"Cloning repository '{REPO_NAME}' into {COLAB_CONTENT_PATH}...")
    result = subprocess.run(["git", "clone", GIT_CLONE_URL, REPO_PATH], capture_output=True, text=True)
    if result.returncode == 0:
        print("Repository cloned successfully.")
        if os.path.isdir(FULL_TRAINER_PATH):
            os.chdir(FULL_TRAINER_PATH)
            print(f"Changed directory to: {os.getcwd()}")
        else:
            print(f"ERROR: Cloned repository but trainer directory '{FULL_TRAINER_PATH}' not found inside.")
            sys.exit("Trainer directory not found post-clone.")
    else:
        print(f"ERROR cloning repository: {result.stderr}")
        sys.exit("Failed to clone repository.")

# --- Install Dependencies ---
print("\nInstalling dependencies from requirements.txt...")
requirements_path = "requirements.txt" # Assumes it's in the current dir (new_model_trainer)
if os.path.exists(requirements_path):
    try:
        # It's good practice to ensure pip is up-to-date first in Colab
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True, capture_output=True, text=True)
        # Install requirements
        result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", requirements_path], check=True, capture_output=True, text=True)
        print("Dependencies installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"ERROR installing dependencies: {e.stderr}")
        if e.stdout: print(f"Stdout: {e.stdout}")
        sys.exit("Failed to install dependencies.")
else:
    print(f"ERROR: {requirements_path} not found in {os.getcwd()}. Cannot install dependencies.")
    sys.exit("requirements.txt not found.")

# --- PyTorch/GPU Check ---
print("\nChecking PyTorch and GPU availability...")
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA is available. GPU: {torch.cuda.get_device_name(0)}")
        # device_to_use_colab = torch.device("cuda") # Will be set in config
    elif torch.backends.mps.is_available():
        print("MPS is available on this device (Apple Silicon GPU).")
        # device_to_use_colab = torch.device("mps")
    else:
        print("No GPU found by PyTorch. Training will use CPU.")
        # device_to_use_colab = torch.device("cpu")
except ImportError:
    print("ERROR: PyTorch is not installed. Please ensure dependencies were installed correctly.")
    # device_to_use_colab = torch.device("cpu") # Fallback
    sys.exit("PyTorch not found after installation attempt.")

# --- Mount Google Drive ---
print("\nMounting Google Drive...")
try:
    from google.colab import drive
    drive.mount('/content/drive', force_remount=True)
    print("Google Drive mounted successfully at /content/drive.")
except ImportError:
    print("Could not import Google Drive module. This notebook is intended for Colab.")
except Exception as e:
    print(f"An error occurred while mounting Google Drive: {e}")

print("\nSetup cell complete.")
```

---
## Cell 3: Markdown
---

## 2. Data Preparation & Configuration File Setup

This section helps you prepare your main JSON configuration file. This file will tell `train_model.py` where to find your audio data and how to set up the training.

---
## Cell 4: Code - Section 2.1: Specify Data and Configuration Location
---

```python
#@markdown ### (RUN CELL) 2.1: Specify Training Configuration File from Google Drive
#@markdown Provide the path to your main data directory on Google Drive and the name of your JSON training configuration file (e.g., `my_run_config.json`) located **within that directory**.

#@markdown **Important:**
#@markdown - Your JSON configuration file should be structured like `new_model_trainer/configs/example_config.json`.
#@markdown - It **must** contain a `dataset_items` list.
#@markdown - Paths within `dataset_items` (i.e., `input_path`, `target_path`) should be **relative to your `DRIVE_DATA_DIR_FOR_CONFIG_AND_AUDIO`** or full paths starting with `/content/drive/MyDrive/...`.
#@markdown - The `output_dir` in your config will be automatically remapped to a local Colab path.

DRIVE_DATA_DIR_FOR_CONFIG_AND_AUDIO = "/content/drive/MyDrive/AidaX_NewTrainer_Data" #@param {type: "string"}
TRAINING_CONFIG_FILENAME_ON_DRIVE = "colab_train_config.json" #@param {type: "string"}

import os
import json
import shutil
import sys # Added for sys.exit

# --- Define local Colab paths ---
COLAB_ROOT_CONFIG_DIR = "/content/colab_run_configs" # Directory to store config files for this run
COLAB_LOCAL_AUDIO_DIR = "/content/colab_run_audio_data"   # Directory to store audio data for this run

# This will be the single config file passed to train_model.py
EFFECTIVE_COLAB_CONFIG_PATH = os.path.join(COLAB_ROOT_CONFIG_DIR, "final_colab_training_config.json")
# Store the original user config path for reference
USER_DRIVE_CONFIG_PATH_ABSOLUTE = os.path.join(DRIVE_DATA_DIR_FOR_CONFIG_AND_AUDIO, TRAINING_CONFIG_FILENAME_ON_DRIVE)


os.makedirs(COLAB_ROOT_CONFIG_DIR, exist_ok=True)
os.makedirs(COLAB_LOCAL_AUDIO_DIR, exist_ok=True)

# Clear out old audio data from previous runs
print(f"Clearing local audio directory: {COLAB_LOCAL_AUDIO_DIR}...")
for item_name in os.listdir(COLAB_LOCAL_AUDIO_DIR):
    item_path_to_delete = os.path.join(COLAB_LOCAL_AUDIO_DIR, item_name)
    try:
        if os.path.isfile(item_path_to_delete) or os.path.islink(item_path_to_delete):
            os.unlink(item_path_to_delete)
        elif os.path.isdir(item_path_to_delete):
            shutil.rmtree(item_path_to_delete)
    except Exception as e:
        print(f'Failed to delete {item_path_to_delete}. Reason: {e}')
print(f"Local audio directory cleared.")


print(f"Attempting to load user's training config from: {USER_DRIVE_CONFIG_PATH_ABSOLUTE}")

config_data_from_drive = None # Initialize
globals()['FINAL_CONFIG_PATH_FOR_TRAINING'] = None # Ensure it's None if steps fail

if not os.path.exists(USER_DRIVE_CONFIG_PATH_ABSOLUTE):
    print(f"ERROR: Main training configuration file not found at: {USER_DRIVE_CONFIG_PATH_ABSOLUTE}")
    print("Please ensure the path and filename are correct and the file exists on your Google Drive.")
    sys.exit("Configuration file not found in Drive.")
else:
    try:
        with open(USER_DRIVE_CONFIG_PATH_ABSOLUTE, 'r') as f:
            config_data_from_drive = json.load(f)
        print("Successfully loaded main configuration file from Drive.")
    except Exception as e:
        print(f"ERROR: Could not parse training configuration file {USER_DRIVE_CONFIG_PATH_ABSOLUTE}. Error: {e}")
        sys.exit("Failed to parse configuration file.")


    if config_data_from_drive:
        if 'dataset_items' not in config_data_from_drive or not isinstance(config_data_from_drive['dataset_items'], list):
            print("ERROR: The configuration file must contain a 'dataset_items' list.")
            sys.exit("Invalid configuration: missing or malformed dataset_items.")
        if not config_data_from_drive['dataset_items']:
            print("ERROR: 'dataset_items' list in configuration is empty. Nothing to process.")
            sys.exit("Empty dataset_items list.")

        updated_dataset_items = []
        all_files_valid = True

        for i, item_config in enumerate(config_data_from_drive['dataset_items']):
            original_input_path_ref = item_config.get('input_path')
            original_target_path_ref = item_config.get('target_path')

            if not original_input_path_ref or not original_target_path_ref:
                print(f"WARNING: Item {i} in config is missing 'input_path' or 'target_path'. Skipping.")
                all_files_valid = False; continue

            full_drive_input_path = original_input_path_ref if original_input_path_ref.startswith("/content/drive/") else os.path.join(DRIVE_DATA_DIR_FOR_CONFIG_AND_AUDIO, original_input_path_ref)
            full_drive_target_path = original_target_path_ref if original_target_path_ref.startswith("/content/drive/") else os.path.join(DRIVE_DATA_DIR_FOR_CONFIG_AND_AUDIO, original_target_path_ref)

            if not os.path.exists(full_drive_input_path):
                print(f"ERROR: Input audio file not found for item {i}: {full_drive_input_path} (Original ref: {original_input_path_ref})")
                all_files_valid = False; continue
            if not os.path.exists(full_drive_target_path):
                print(f"ERROR: Target audio file not found for item {i}: {full_drive_target_path} (Original ref: {original_target_path_ref})")
                all_files_valid = False; continue

            input_basename = os.path.basename(original_input_path_ref)
            target_basename = os.path.basename(original_target_path_ref)
            local_input_filename = f"{i:03d}_input_{input_basename}"
            local_target_filename = f"{i:03d}_target_{target_basename}"

            local_colab_input_path = os.path.join(COLAB_LOCAL_AUDIO_DIR, local_input_filename)
            local_colab_target_path = os.path.join(COLAB_LOCAL_AUDIO_DIR, local_target_filename)

            try:
                shutil.copy(full_drive_input_path, local_colab_input_path)
                shutil.copy(full_drive_target_path, local_colab_target_path)

                updated_item_config = item_config.copy()
                updated_item_config['input_path'] = local_colab_input_path
                updated_item_config['target_path'] = local_colab_target_path
                updated_dataset_items.append(updated_item_config)
            except Exception as e:
                print(f"ERROR: Failed to copy files for item {i}. Input: {full_drive_input_path}. Error: {e}")
                all_files_valid = False; continue

        if not all_files_valid:
             print("ERROR: Some audio files could not be processed. Check paths and file existence. Cannot proceed.")
             sys.exit("Audio file processing failed.")
        elif not updated_dataset_items: # Should be caught by previous if all_files_valid is false, but as a safeguard
            print("ERROR: No dataset items were successfully processed from the config. Cannot proceed.")
            sys.exit("Dataset item processing resulted in no usable items.")
        else:
            config_data_from_drive['dataset_items'] = updated_dataset_items

            model_name_from_config = config_data_from_drive.get('model_name', 'default_model')
            model_name_sanitized = "".join(c if c.isalnum() else "_" for c in model_name_from_config)

            local_colab_output_dir = f"/content/training_output/{model_name_sanitized}"
            config_data_from_drive['output_dir'] = local_colab_output_dir
            os.makedirs(local_colab_output_dir, exist_ok=True)
            print(f"Output for this run will be in local Colab directory: {local_colab_output_dir}")

            with open(EFFECTIVE_COLAB_CONFIG_PATH, 'w') as f:
                json.dump(config_data_from_drive, f, indent=4)
            print(f"Effective training configuration with local paths saved to: {EFFECTIVE_COLAB_CONFIG_PATH}")
            globals()['FINAL_CONFIG_PATH_FOR_TRAINING'] = EFFECTIVE_COLAB_CONFIG_PATH
            print("\nData preparation and localization complete.")

if not globals().get('FINAL_CONFIG_PATH_FOR_TRAINING'):
    print("\nERROR: Data preparation failed. Cannot proceed to training configuration overrides.")
```

---
## Cell 5: Markdown
---

## 3. Training Configuration Overrides (Optional)

Adjust common training parameters here if you want to quickly experiment without editing your main JSON configuration file. Values from your JSON file will be used if you don't specify an override here.

---
## Cell 6: Code - Section 3.1: Adjust Training Parameters
---

```python
#@markdown ### (RUN CELL) 3.1: Adjust Training Parameters (Optional Overrides)
#@markdown Leave fields as their default (e.g., empty string, -1, 0.0, "AUTO") to use the value from your JSON config loaded in the previous step.

#@markdown ---
#@markdown **Basic Settings:**
MODEL_NAME_OVERRIDE = "" #@param {type:"string"}
#@markdown Number of epochs (-1 to use config value).
EPOCHS_OVERRIDE = -1 #@param {type:"integer"}
#@markdown Batch size (-1 to use config value).
BATCH_SIZE_OVERRIDE = -1 #@param {type:"integer"}
#@markdown Learning rate (e.g., 0.001; 0.0 to use config value).
LEARNING_RATE_OVERRIDE = 0.0 #@param {type:"number"}
#@markdown Device: "AUTO" (use config), "cuda", "cpu", "mps".
DEVICE_OVERRIDE = "AUTO" #@param ["AUTO", "cuda", "cpu", "mps"]

#@markdown ---
#@markdown **Model Architecture (Overrides if specified):**
#@markdown LSTM Hidden Size (-1 for config value).
HIDDEN_SIZE_OVERRIDE = -1 #@param {type:"integer"}
#@markdown Number of LSTM layers (-1 for config value).
NUM_LAYERS_OVERRIDE = -1 #@param {type:"integer"}
#@markdown Skip Connection: "AUTO" (use config), "ON", "OFF".
SKIP_CONNECTION_OVERRIDE = "AUTO" #@param ["AUTO", "ON", "OFF"]

#@markdown ---
#@markdown **Advanced (Overrides if specified):**
#@markdown Early stopping patience (-1 for config value).
EARLY_STOPPING_PATIENCE_OVERRIDE = -1 #@param {type:"integer"}

import json
import os
import sys

print("Applying Colab form overrides to the training configuration...")

# Path to the configuration file prepared by the Data Preparation cell
# FINAL_CONFIG_PATH_FOR_TRAINING should have been set by the previous cell.
current_config_path = globals().get('FINAL_CONFIG_PATH_FOR_TRAINING')

if not current_config_path or not os.path.exists(current_config_path):
    print(f"ERROR: Effective configuration file not found at '{current_config_path}' (or not set).")
    print("Please ensure the 'Data Preparation' cell (2.1) was run successfully and resulted in a valid config path.")
    sys.exit("Effective config not found, cannot apply overrides.")
else:
    with open(current_config_path, 'r') as f:
        config_data = json.load(f)
    print(f"Loaded configuration from: {current_config_path} to apply overrides.")

    # Apply overrides from Colab form parameters
    if MODEL_NAME_OVERRIDE:
        original_model_name = config_data.get('model_name', 'default_model')
        config_data['model_name'] = MODEL_NAME_OVERRIDE
        # If model name changes, output_dir should also change to reflect the new name
        new_model_name_sanitized = "".join(c if c.isalnum() else "_" for c in MODEL_NAME_OVERRIDE)
        config_data['output_dir'] = f"/content/training_output/{new_model_name_sanitized}"
        os.makedirs(config_data['output_dir'], exist_ok=True) # Ensure it exists
        print(f"Overridden model_name to: {MODEL_NAME_OVERRIDE}. Output directory updated to: {config_data['output_dir']}")

    if EPOCHS_OVERRIDE != -1:
        config_data['epochs'] = EPOCHS_OVERRIDE
        print(f"Overridden epochs to: {EPOCHS_OVERRIDE}")
    if BATCH_SIZE_OVERRIDE != -1:
        config_data['batch_size'] = BATCH_SIZE_OVERRIDE
        print(f"Overridden batch_size to: {BATCH_SIZE_OVERRIDE}")
    if LEARNING_RATE_OVERRIDE != 0.0:
        config_data['learning_rate'] = LEARNING_RATE_OVERRIDE
        print(f"Overridden learning_rate to: {LEARNING_RATE_OVERRIDE}")
    if DEVICE_OVERRIDE != "AUTO":
        config_data['device'] = DEVICE_OVERRIDE
        print(f"Overridden device to: {DEVICE_OVERRIDE}")

    if HIDDEN_SIZE_OVERRIDE != -1:
        config_data['hidden_size'] = HIDDEN_SIZE_OVERRIDE
        print(f"Overridden hidden_size to: {HIDDEN_SIZE_OVERRIDE}")
    if NUM_LAYERS_OVERRIDE != -1:
        config_data['num_layers'] = NUM_LAYERS_OVERRIDE
        print(f"Overridden num_layers to: {NUM_LAYERS_OVERRIDE}")
    if SKIP_CONNECTION_OVERRIDE != "AUTO":
        config_data['skip_connection'] = True if SKIP_CONNECTION_OVERRIDE == "ON" else False
        print(f"Overridden skip_connection to: {config_data['skip_connection']}")
    if EARLY_STOPPING_PATIENCE_OVERRIDE != -1:
        config_data['early_stopping_patience'] = EARLY_STOPPING_PATIENCE_OVERRIDE
        print(f"Overridden early_stopping_patience to: {EARLY_STOPPING_PATIENCE_OVERRIDE}")

    # Save the potentially modified configuration back (overwriting the previous "effective" config)
    with open(current_config_path, 'w') as f:
        json.dump(config_data, f, indent=4)
    print(f"Final configuration with overrides applied saved to: {current_config_path}")
    print(f"Final model name for this run: {config_data.get('model_name', 'N/A')}")
    print(f"Output directory for this run: {config_data.get('output_dir', 'N/A')}")
    print("\nTraining Configuration cell complete.")

    # Ensure the global variable still points to the (now possibly modified) config file
    globals()['FINAL_CONFIG_PATH_FOR_TRAINING'] = current_config_path
```

---
## Cell 7: Markdown
---

## 4. Run Training

This cell executes the main training script `train_model.py` using the configuration file prepared and potentially modified in the steps above.

---
## Cell 8: Code - Section 4.1: Start Model Training
---

```python
#@markdown ### (RUN CELL) 4.1: Start Model Training
#@markdown This cell will execute the `train_model.py` script. Training progress will be displayed below.
#@markdown This may take a significant amount of time depending on your data and settings.

import os
import json # For reading the final config to show output_dir
import sys # For sys.exit

# FINAL_CONFIG_PATH_FOR_TRAINING should have been set by the previous cells.
final_config_to_use = globals().get('FINAL_CONFIG_PATH_FOR_TRAINING')

if not final_config_to_use or not os.path.exists(final_config_to_use):
    print(f"ERROR: Final configuration file '{final_config_to_use}' not found or not set!")
    print("Please ensure the 'Data Preparation' (2.1) and 'Training Configuration' (3.1) cells were run successfully.")
    sys.exit("Final config not found. Cannot start training.")
else:
    print(f"Starting training using configuration: {final_config_to_use}")
    # FULL_TRAINER_PATH from setup cell should be the current working directory
    # If not, os.chdir(FULL_TRAINER_PATH) might be needed if setup cell didn't persist or was skipped.
    print(f"Current working directory is: {os.getcwd()}. Expecting this to be the 'new_model_trainer' directory.")

    # Execute the training script
    # Using get_ipython().system() to run shell commands in Colab and see output
    get_ipython().system(f'python train_model.py --config_file="{final_config_to_use}"')

    print("\n--- Training Run Attempted ---")
    print("If training started, check the output above for progress and completion status.")

    # Try to load the config again to find the output directory and model name
    try:
        with open(final_config_to_use, 'r') as f:
            final_config_data_check = json.load(f)
        output_dir_from_config_check = final_config_data_check.get('output_dir')
        model_name_from_config_check = final_config_data_check.get('model_name')

        if output_dir_from_config_check and os.path.isdir(output_dir_from_config_check):
            print(f"\nContents of the output directory ({output_dir_from_config_check}):")
            for item_name_check in os.listdir(output_dir_from_config_check):
                print(f"- {item_name_check}")

            if model_name_from_config_check:
                expected_aidax_path_check = os.path.join(output_dir_from_config_check, f"{model_name_from_config_check}.aidax")
                if os.path.exists(expected_aidax_path_check):
                    print(f"\nSUCCESS: Exported model likely found at {expected_aidax_path_check}")
                else:
                    print(f"\nNOTE: Exported .aidax model not immediately found at {expected_aidax_path_check}. Training/export might have failed or check subdirectories.")
            else:
                print("\nNOTE: 'model_name' not found in config, cannot construct expected .aidax path.")
        elif output_dir_from_config_check: # If output_dir was specified but doesn't exist
             print(f"\nNOTE: Output directory {output_dir_from_config_check} (specified in config) was NOT found after training attempt.")
        else: # If output_dir was not in config for some reason
            print("\nNOTE: Could not determine 'output_dir' from config to list results.")
    except Exception as e:
        print(f"Note: Could not list output directory contents due to an error: {e}")
```

---
## Cell 9: Markdown
---

## 5. Export & Download Model

The `train_model.py` script should automatically export the best model to an `.aidax` file in its output directory. This section helps you locate that file and provides options to download it or copy it to your Google Drive.

---
## Cell 10: Code - Section 5.1: Access Your Trained Model
---

```python
#@markdown ### (RUN CELL) 5.1: Access Your Trained Model
#@markdown This cell helps you locate the exported `.aidax` model file and provides options to download it or copy it to your Google Drive.

#@markdown ---
#@markdown **Specify Google Drive Destination (Optional):**
#@markdown If you want to copy the `.aidax` file to a specific folder in your Google Drive, enter the path below relative to your "My Drive" root.
#@markdown Example: `AidaX_TrainedModels/` (this will be placed in `/content/drive/MyDrive/AidaX_TrainedModels/`).
#@markdown Ensure the base path (`/content/drive/MyDrive/`) exists from mounting.
GDRIVE_RELATIVE_EXPORT_PATH = "AidaX_TrainedModels_Output" #@param {type:"string"}

import os
import json
import shutil
from google.colab import files # For downloading
import sys # For sys.exit

print("Locating the exported model...")
# FINAL_CONFIG_PATH_FOR_TRAINING should have been set by previous cells
final_config_to_use_for_download = globals().get('FINAL_CONFIG_PATH_FOR_TRAINING')

if not final_config_to_use_for_download or not os.path.exists(final_config_to_use_for_download):
    print(f"ERROR: Final configuration file '{final_config_to_use_for_download}' not found or not set!")
    print("Please ensure the entire notebook has been run successfully up to the training step.")
    sys.exit("Final config not found. Cannot locate model.")
else:
    try:
        with open(final_config_to_use_for_download, 'r') as f:
            config_data_for_download = json.load(f)

        model_name_for_download = config_data_for_download.get('model_name')
        # This is the local Colab output directory, e.g., /content/training_output/MODEL_NAME
        local_colab_output_dir_for_download = config_data_for_download.get('output_dir')

        if not model_name_for_download or not local_colab_output_dir_for_download:
            print("ERROR: 'model_name' or 'output_dir' not found in the configuration. Cannot locate .aidax file.")
            sys.exit("Missing critical config info for locating model.")
        elif not os.path.isdir(local_colab_output_dir_for_download):
            print(f"ERROR: The output directory '{local_colab_output_dir_for_download}' specified in the config does not exist. Training may have failed or not run.")
            sys.exit("Output directory not found.")
        else:
            expected_aidax_filename_download = f"{model_name_for_download}.aidax"
            local_aidax_path_download = os.path.join(local_colab_output_dir_for_download, expected_aidax_filename_download)

            if os.path.exists(local_aidax_path_download):
                print(f"SUCCESS: Exported model found at local Colab path: {local_aidax_path_download}")
                file_size_kb = os.path.getsize(local_aidax_path_download) / 1024
                print(f"File size: {file_size_kb:.2f} KB")

                # Option 1: Download the file
                print("\n--- Option 1: Download to your local machine ---")
                try:
                    files.download(local_aidax_path_download)
                    print(f"Download initiated for {expected_aidax_filename_download}.")
                except NameError: # If files module not available (e.g. not in Colab)
                     print("Could not initiate download (google.colab.files not available).")
                except Exception as e:
                    print(f"Could not initiate download. Error: {e}")
                print(f"If download doesn't start automatically, you can manually download from the Colab file browser (left panel) at: {local_aidax_path_download}")


                # Option 2: Copy to Google Drive
                if GDRIVE_RELATIVE_EXPORT_PATH:
                    # Sanitize relative path to prevent '..' etc.
                    gdrive_safe_relative_path = os.path.normpath(GDRIVE_RELATIVE_EXPORT_PATH.strip('/'))
                    if '..' in gdrive_safe_relative_path.split(os.path.sep) or gdrive_safe_relative_path.startswith('/'):
                         print(f"ERROR: Invalid Google Drive relative path: '{GDRIVE_RELATIVE_EXPORT_PATH}'. It should not contain '..' or start with '/'.")
                    else:
                        full_gdrive_export_dir = os.path.join("/content/drive/MyDrive/", gdrive_safe_relative_path)
                        print(f"\n--- Option 2: Copy to Google Drive ---")
                        print(f"Attempting to copy to: {full_gdrive_export_dir}")

                        try:
                            if not os.path.exists(full_gdrive_export_dir):
                                print(f"Google Drive destination path '{full_gdrive_export_dir}' does not exist. Attempting to create it.")
                                os.makedirs(full_gdrive_export_dir, exist_ok=True)
                                print(f"Created Google Drive directory: {full_gdrive_export_dir}")

                            drive_destination_file_path = os.path.join(full_gdrive_export_dir, expected_aidax_filename_download)
                            shutil.copy(local_aidax_path_download, drive_destination_file_path)
                            print(f"Model successfully copied to Google Drive: {drive_destination_file_path}")
                        except Exception as e:
                            print(f"ERROR: Failed to copy model to Google Drive path '{full_gdrive_export_dir}'. Error: {e}")
                else:
                    print("\n--- Option 2: Copy to Google Drive ---")
                    print("No Google Drive destination path specified by user. Skipping copy to Drive.")
            else:
                print(f"ERROR: Exported model file not found at expected local Colab location: {local_aidax_path_download}")
                print("This might indicate that the training or export step failed or was not completed.")
                print(f"Please check the output of the 'Run Training' cell and the contents of '{local_colab_output_dir_for_download}'")

    except Exception as e:
        print(f"An error occurred while trying to access the model for export/download: {e}")
        print("Ensure previous cells, especially training, completed successfully and a config path was established.")

print("\nExport & Download Model cell execution finished.")
```

---
## Cell 11: Markdown (Optional - for Evaluation)
---

## 6. Model Evaluation (Optional)

This section can be used to load your trained PyTorch model (from the `.pth` checkpoint in your local Colab `output_dir`) and perform some basic evaluation, like plotting predictions against target audio or listening to samples.
(Implementation of this section is left as a future exercise if needed, as it would require loading data again and running model inference within this notebook).

```
