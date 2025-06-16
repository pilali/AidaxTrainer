# Creating a valid dataset for the trainining script
# using wav files provided by user.
# Example of usage:
# python3 prep_wav.py -f input.wav target.wav -l "RNN-aidadsp-1"
# the files will be splitted 70% 15% 15%
# and used to populate train test val.
# This is done to have different data for training, testing and validation phase
# according with the paper.
# If the user provide multiple wav files pairs e.g. guitar_in.wav guitar_tg.wav bass_in.wav bass_tg.wav
# then 70% of guitar_in.wav is concatenated to 70% of bass_in.wav and so on.
# If the user provide guitar and bass files of the same length, then the same amount
# of guitar and bass recorded material will be used for network training.

from CoreAudioML.dataset import audio_splitter
from scipy.io.wavfile import write as wavfilewrite
from scipy.signal import spectrogram
from scipy.signal import savgol_filter
import numpy as np
from torch import tensor as torchtensor
from torch import no_grad as torchnograd
import time
import os
import csv
import librosa
import json
import argparse

import PIL.Image
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import io

def smoothed_spectrogram(x, fs=48000, window="hann", size=4096, mode='peak'):
    '''
    Calculate peak spectrogram
    - x: numpy array, time series expected ndim=1
    - mode: peak or avg
    '''
    if x.ndim < 1 or x.ndim > 1:
        print("Error: unsupported dimension for input x!")
    N = size
    f, t, Sxx = spectrogram(x, fs=fs, window=window, nperseg=N, mode='magnitude')
    Sxx_split = np.array_split(Sxx, np.size(f))
    if mode == 'avg':
        Sxx_avg = [np.mean(arr) for arr in Sxx_split]
        Sxx_avg_dB = 10.0 * np.log10(Sxx_avg)
        Sxx_avg_dB_smoothed = savgol_filter(Sxx_avg_dB, N//10, 3)
        return f, Sxx_avg_dB_smoothed, np.min(Sxx_avg_dB_smoothed), np.max(Sxx_avg_dB_smoothed)
    else:
        Sxx_peak = [np.max(arr) for arr in Sxx_split]
        Sxx_peak_dB = 10.0 * np.log10(Sxx_peak)
        Sxx_peak_dB_smoothed = savgol_filter(Sxx_peak_dB, N//10, 3)
        return f, Sxx_peak_dB_smoothed, np.min(Sxx_peak_dB_smoothed), np.max(Sxx_peak_dB_smoothed)

def gen_smoothed_spectrogram_plot(f=None, target=None, predicted=None, title=''):
    plt.figure()
    if target is not None:
        plt.semilogx(f, target, 'b-', label="Target")
    plt.semilogx(f, predicted, 'r-', label="Predicted")
    plt.grid()
    plt.xlabel("Hz")
    plt.ylabel("dB")
    plt.title(title)
    plt.legend()
    return plt

def pyplot_to_tensor(plt=None):
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    img = PIL.Image.open(buf)
    return ToTensor()(img).unsqueeze(0)[0]

# WARNING! De-noise is currently experimental and just for research / documentation
def denoise(method="noisereduce", waveform=np.ndarray([0], dtype=np.float32), noise_locations=(0, 6_000), samplerate=48000):
    import noisereduce as nr
    from CoreAudioML.training import ESRLoss

    noise = waveform[noise_locations[0]:noise_locations[1]]
    print("Noise level: %.6f [dBTp]" % peak(noise))

    if method == "noisereduce":
        denoise = nr.reduce_noise(y=waveform, sr=samplerate, y_noise=noise, n_std_thresh_stationary=1.5, stationary=True, prop_decrease=1.0, n_fft=2048, n_jobs=-1)
    elif method == "simplefilter":
        denoise = apply_filter(waveform=waveform, samplerate=samplerate)
    waveform = denoise

    noise = waveform[noise_locations[0]:noise_locations[1]]
    print("Noise level after denoise: %.6f [dBTp]" % peak(noise))

    return denoise

# Calculate lowest theoretical ESR after denoise
# this is feasible only if a section, typically val or test is repeated accross the Dataset.
# NOTE: ESR is calculated without pre-emphasis filter
def calculate_min_theoretical_esr_loss(waveform, locations=(8160000, 8592000, 8592000, 9024000), samplerate: int = 48000):
    val1_t = torchtensor(waveform[locations[0]:locations[1]])
    val2_t = torchtensor(waveform[locations[2]:locations[3]])
    lossESR = ESRLoss()
    ESRmin = lossESR(val1_t, val2_t)
    print("Min theoretical ESR is %.6f" % ESRmin)
    return ESRmin

# Apply a filter using torchaudio.functional
def apply_filter(filter_type='highpass', waveform=None, samplerate: int = 48000, frequency=120.0, Q=0.707):
    try:
        if len(waveform) == 0:
            print("Error: no data to process")
            exit(1)
    except TypeError:
        exit(1)
    waveform = torchtensor(waveform)
    if waveform.dim() != 1:
        print("Error: expected dim = 1, but it's %d" % waveform.dim())
        exit(1)
    if filter_type == 'highpass':
        from torchaudio.functional import highpass_biquad as hp
        out = hp(waveform=waveform, sample_rate=samplerate, cutoff_freq=frequency, Q=Q)
    elif filter_type == 'lowpass':
        from torchaudio.functional import lowpass_biquad as lp
        out = lp(waveform=waveform, sample_rate=samplerate, cutoff_freq=frequency, Q=Q)
    elif filter_type == 'bandpass':
        from torchaudio.functional import bandpass_biquad as bp
        out = bp(waveform=waveform, sample_rate=samplerate, central_freq=frequency, Q=Q)
    return out.cpu().data.numpy()

# This creates a csv file containing regions for input.wav proposed by current public
# release of AIDA-X. This file is longer than NAM Dataset, containing human-played dry
# guitar riffs. According to our experiments a longer Dataset usually improves the final model.
# The content of this file follows Reaper region markers export csv format
def create_csv_aidax(path):
    header = ['#', 'Name', 'Start', 'End', 'Length', 'Color']
    data = [
        ['R1','samplerate','0','48000','48000','FFFF00'],
        ['R2', 'noise', '0', '6000', '6000', 'FFFF00'],
        ['R3', 'blips', '12000', '36000', '24000', 'FFFF00'],
        ['R4', 'nam_train', '50000', '8160000', '8110000', 'FF0000'],
        ['R5', 'nam_val+test', '8160000', '8592000', '432000', '00FFFF'],
        ['R6', '_train', '8592000', '14523000', '5931000', 'FF0000'],
        ['R7', 'end', '14523000', '14523032', '32', 'FFFF00']
    ]
    with open(path, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)

# This creates a csv file containing regions for NAM v1_1_1.wav and leaved as reference.
# The content of this file follows Reaper region markers export csv format
def create_csv_nam_v1_1_1(path):
    header = ['#', 'Name', 'Start', 'End', 'Length', 'Color']
    data = [
        ['R1','samplerate','0','48000','48000','FFFF00'],
        ['R2', 'noise', '0', '6000', '6000', 'FFFF00'],
        ['R3', 'blips', '12000', '36000', '24000', 'FFFF00'],
        ['R4', 'nam_train', '50000', '8160000', '8110000', 'FF0000'],
        ['R5', 'nam_val+test', '8160000', '8592000', '432000', '00FFFF'],
        ['R6', 'end', '8592000', '8592032', '32', 'FFFF00']
    ]
    with open(path, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)

# This creates a csv file containing regions for NAM v2_0_0.wav and leaved as reference.
# The content of this file follows Reaper region markers export csv format
def create_csv_nam_v2_0_0(path):
    header = ['#', 'Name', 'Start', 'End', 'Length', 'Color']
    data = [
        ['R1','samplerate','0','48000','48000','FFFF00'],
        ['R2', 'noise', '12000', '18000', '6000', 'FFFF00'],
        ['R3', 'blips', '24000', '72000', '48000', 'FFFF00'],
        ['R4', 'nam_train', '90000', '8208000', '8118000', 'FF0000'],
        ['R5', 'nam_val+test', '8208000', '8640000', '432000', '00FFFF'],
        ['R6', 'nam_val_', '8640000', '9072000', '432000', 'FFFF00'],
        ['R7', 'blips_', '9096000', '9144000', '48000', 'FFFF00'],
        ['R8', 'end', '9168000', '9168032', '32', 'FFFF00']
    ]
    with open(path, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)

def peak(data, target=None):
    """
    Based on pyloudnorm from https://github.com/csteinmetz1/pyloudnorm/blob/1fb914693e07f4bea06fdb2e4bd2d6ddc1688a9e/pyloudnorm/normalize.py#L5
    Copyright (c) 2021 Steinmetz, Christian J. and Reiss, Joshua D.
    SPDX - License - Identifier: MIT
    """
    """ Peak normalize a signal.

    Normalize an input signal to a user specifed peak amplitude.
    Params
    -------
    data : ndarray
        Input multichannel audio data.
    target : float
        Desired peak amplitude in dB. If not provided, return
    Returns
    -------
    output : ndarray
        Peak normalized output data.
    """
    # find the amplitude of the largest peak
    current_peak = np.max(np.abs(data))

    # if no target is provided, it's a measure
    if not target:
        return np.multiply(20.0, np.log10(current_peak))

    # calculate the gain needed to scale to the desired peak level
    gain = np.power(10.0, target/20.0) / current_peak
    output = gain * data

    # check for potentially clipped samples
    if np.max(np.abs(output)) >= 1.0:
        print("Possible clipped samples in output.")

    return output

def wav2tensor(filepath):
  aud, sr = librosa.load(filepath, sr=None, mono=True)
  aud = librosa.resample(aud, orig_sr=sr, target_sr=48000)
  return torchtensor(aud)

def extract_best_esr_model(dirpath):
  stats_file = dirpath + "/training_stats.json"
  with open(stats_file) as json_file:
    stats_data = json.load(json_file)
    test_lossESR_final = stats_data['test_lossESR_final']
    test_lossESR_best = stats_data['test_lossESR_best']
    esr = min(test_lossESR_final, test_lossESR_best)
    if esr == test_lossESR_final:
      model_path = dirpath + "/model.json"
    else:
      model_path = dirpath + "/model_best.json"
  return model_path, esr

def save_wav(name, rate, data, flatten=True):
    # print("Writing %s with rate: %d length: %d dtype: %s" % (name, rate, data.size, data.dtype))
    if flatten:
        wavfilewrite(name, rate, data.flatten().astype(np.float32))
    else:
        wavfilewrite(name, rate, data.astype(np.float32))

def shift_info(info, shift: int = 0):
    new_info = {}
    for key, values in info.items():
        if isinstance(values, list):
            new_info[key] = [(v[0] + shift, v[1] + shift) for v in values]
        else:
            # Handle other types if necessary
            pass
    return new_info

def scale_info(info, scale_factor: float = 1.0):
    scaled_info = {}
    for key, values in info.items():
        if isinstance(values, list):
            scaled_info[key] = [(int(v[0] * scale_factor), int(v[1] * scale_factor)) for v in values]
        else:
            # Handle other types if necessary
            pass
    return scaled_info

def convert_csv_to_info(csv_path):
    info = {}
    with open(csv_path, 'r', encoding='UTF8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            tag, name, start, end, length, color = row
            if name not in info:
                info[name] = []
            info[name].append((int(start), int(end)))
    return info

def convert_info_to_csv(info):
    header = ['#', 'Name', 'Start', 'End', 'Length', 'Color']
    data = []
    counter = 1
    for key, values in info.items():
        if isinstance(values, list):
            for value in values:
                tag = "R%d" % counter
                name = key
                start, end = value
                length = end - start
                color = 'FFFF00'  # Pick a color
                data.append([tag, name, start, end, length, color])
                counter += 1
        else:
            # Handle other types if necessary
            pass
    return header, data

def save_csv(path, info):
    header, data = convert_info_to_csv(info)

    with open(path, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)

def parse_info(info):
    train_bounds = []
    test_bounds = []
    val_bounds = []
    for key, values in info.items():
        if isinstance(values, list):
            for value in values:
                if key.endswith("_train"):
                    train_bounds.append(value)
                elif key.endswith("_test"):
                    test_bounds.append(value)
                elif key.endswith("_val"):
                    val_bounds.append(value)
                elif key.endswith("_val+test"):
                    test_bounds.append(value)
                    val_bounds.append(value)
        else:
            # Handle other types if necessary
            pass

    if len(train_bounds) < 1 or len(test_bounds) < 1 or len(val_bounds) < 1:
        print("Error: info does not contain all necessary keys")
        exit(1)

    return[train_bounds, test_bounds, val_bounds]

# This method deducts the samplerate from the corresponding samplerate key in the info dictionary
def get_info_samplerate(info):
    try:
        samplerate = int(info['samplerate'][0][1])
    except KeyError:
        print("Error: samplerate not found in info")
        exit(1)
    return samplerate

def extract_audio_tag(in_file, path_csv, tag=''):
    """
    Extract audio bounds corresponding to tag occurences in a csv file and return them as numpy.ndarray
    """
    in_data, in_rate = librosa.load(in_file, sr=None, mono=True)
    in_data = librosa.resample(in_data, orig_sr=in_rate, target_sr=in_rate)
    bounds = []
    with open(path_csv) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count != 0:
                if row[1] == tag:
                    bounds.append([int(row[2]), int(row[3])])
            line_count = line_count + 1
    out = np.ndarray([0], dtype=np.float32)

    for bounds in bounds:
        out = np.append(out, audio_splitter(in_data, bounds, unit='s'))
    return out

def bounds_from_csv(path_csv, tag=''):
    """
    Extract bounds corresponding to tag occurences in a csv file
    """
    bounds = []
    with open(path_csv) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count != 0:
                if row[1] == tag:
                    bounds.append([int(row[2]), int(row[3])])
            line_count = line_count + 1
    return bounds

def process_wav_with_clipping(input_wav_path, clipper, output_suffix="_processed", device=None, dtype=None, rate=48000):
    """
    Process a WAV file using a clipping implementation and save the processed output.
    Args:
        input_wav_path (str): Path to the input WAV file.
        clipper (torch.nn.Module): Clipping module (e.g., StandardCubicClip, AdvancedClip).
        output_suffix (str): Suffix to append to the output file name.
        device (str or torch.device): Device to use for processing (e.g., "cuda" or "cpu").
        dtype (torch.dtype): Data type for tensors.
        rate (int): Target sample rate for processing.
    """
    factory_kwargs = {'device': device, 'dtype': dtype}

    # Load WAV file and convert to numpy
    audio, sr = librosa.load(input_wav_path, sr=None, mono=True)
    audio = librosa.resample(audio, orig_sr=sr, target_sr=rate)  # Resample to the specified rate

    # Convert to tensor and allocate to the specified device and dtype
    audio_tensor = torchtensor(audio, **factory_kwargs)

    # Ensure the clipper is on the same device and dtype
    clipper = clipper.to(**factory_kwargs)

    # Perform clipping
    with torchnograd():  # Disable gradient computation for inference
        processed_audio_tensor = clipper(audio_tensor)

    # Retrieve output and convert back to numpy
    processed_audio = processed_audio_tensor.cpu().numpy()

    # Save the processed audio to a new WAV file
    output_wav_path = os.path.splitext(input_wav_path)[0] + output_suffix + ".wav"
    save_wav(output_wav_path, rate=rate, data=processed_audio)

    print(f"Processed file saved to: {output_wav_path}")
