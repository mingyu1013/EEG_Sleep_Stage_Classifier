import os
import re
import warnings
import numpy as np
import mne
import torch
import torch.nn.functional as F
from scipy.signal import firwin
from numpy.lib.stride_tricks import as_strided
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

SLEEP_MAP = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4
}

# FIR kernel design
def design_kernel(low, high, fs, taps=129):
    coeffs = firwin(taps, [low, high], fs=fs, pass_zero=False)
    return torch.tensor(coeffs, dtype=torch.float32, device=device)[None, None, :]

# Vectorized sliding windows
def sliding_windows(x, win_len, stride):
    n = x.shape[0]
    num = (n - win_len) // stride + 1
    if num <= 0:
        return np.empty((0, win_len), dtype=x.dtype)
    shape = (num, win_len)
    strides = (x.strides[0] * stride, x.strides[0])
    return as_strided(x, shape=shape, strides=strides)

# preprocess one session
def preprocess_session(psg_path, hyp_path, fs=100, win_sec=30, stride_sec=15):
    raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)
    raw.pick_channels(["EEG Pz-Oz"])
    ann = mne.read_annotations(hyp_path)
    ann.crop(0, raw.times[-1])
    raw.set_annotations(ann)
    raw.filter(0.5, 30.0, fir_design="firwin", verbose=False)
    raw.resample(fs, npad="auto")

    events, _ = mne.events_from_annotations(raw, event_id=SLEEP_MAP, verbose=False)
    valid = events[np.isin(events[:, 2], list(SLEEP_MAP.values()))]
    if len(valid) == 0:
        return None

    total_time = raw.n_times / raw.info['sfreq']
    tmax = min(win_sec - 1.0/fs, total_time - 1.0/fs)
    if tmax <= 0:
        return None

    try:
        epochs = mne.Epochs(raw, valid, tmin=0.0, tmax=tmax,
                            baseline=None, detrend=1,
                            preload=True, verbose=False)
    except ValueError:
        return None

    data = epochs.get_data()[:, 0, :]
    labels = epochs.events[:, 2]

    kernel = design_kernel(0.5, 30.0, fs)
    pad = kernel.shape[-1] // 2
    win_len = int(win_sec * fs)
    stride = int(stride_sec * fs)

    X_list, y_list, c_list = [], [], []
    for sig, lab in zip(data, labels):
        tensor = torch.from_numpy(sig.astype(np.float32)).to(device)[None, None, :]
        filtered = F.conv1d(tensor, kernel, padding=pad).squeeze().cpu().numpy()

        segments = sliding_windows(filtered, win_len, stride)
        if segments.size == 0:
            continue

        freqs = np.fft.rfftfreq(filtered.size, d=1/fs)
        psd = np.abs(np.fft.rfft(filtered))**2
        peak_idx = np.argmax(psd[1:]) + 1
        peak_freq = freqs[peak_idx] if peak_idx < len(freqs) else 0
        cycle_val = fs / peak_freq if peak_freq > 0 else 0

        X_list.append(segments)
        y_list.extend([lab] * segments.shape[0])
        c_list.extend([cycle_val] * segments.shape[0])

    if not X_list:
        return None

    X = np.vstack(X_list)
    y = np.array(y_list)
    c = np.array(c_list)
    return X, y, c

# process all directories
def run_all(psg_dirs, out_path):
    for d in psg_dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"Directory not found: {d}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    total = processed = skipped = 0
    pattern = re.compile(r'^([A-Za-z]{2}\d{4})')
    all_X = all_y = all_c = None

    for d in psg_dirs:
        print(f"Processing {d}")
        files = sorted(f for f in os.listdir(d) if f.endswith("-PSG.edf"))
        for fname in tqdm(files, desc="Sessions"):
            total += 1
            m = pattern.match(fname)
            if not m:
                skipped += 1
                continue
            key = m.group(1)
            hyp_files = [h for h in os.listdir(d)
                         if h.startswith(key) and h.endswith("-Hypnogram.edf")]
            if not hyp_files:
                skipped += 1
                continue

            psg_path = os.path.join(d, fname)
            hyp_path = os.path.join(d, hyp_files[0])
            res = preprocess_session(psg_path, hyp_path)
            if res is None:
                skipped += 1
                continue
            X, y, c = res
            if all_X is None:
                all_X, all_y, all_c = X, y, c
            else:
                all_X = np.vstack((all_X, X))
                all_y = np.concatenate((all_y, y))
                all_c = np.concatenate((all_c, c))
            processed += 1

    if processed == 0:
        raise RuntimeError(f"No sessions processed: total={total}, skipped={skipped}")

    np.savez_compressed(out_path, X_window=all_X, y_stage=all_y, cycle_stl=all_c)
    print(f"Completed: total={total}, processed={processed}, skipped={skipped}")
    print(f"Saved: {all_X.shape[0]} samples to {out_path}")

if __name__ == "__main__":
    PSG_DIRS = [
        "/home/cymg0001/sleep/EDF/sleep-edf-database-expanded-1.0.0/sleep-cassette",
        "/home/cymg0001/sleep/EDF/sleep-edf-database-expanded-1.0.0/sleep-telemetry"
    ]
    OUT_PATH = "/home/cymg0001/sleep/preprocessing_lstm/preprocessed_gpu.npz"
    run_all(PSG_DIRS, OUT_PATH)
