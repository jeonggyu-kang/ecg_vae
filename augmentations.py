
import numpy as np
import scipy.fft   as scift
import scipy.stats as scistat
import matplotlib.pyplot as plt
import pickle
import copy
import gzip

def motion_artifact(signal, fs):
    """
    "Modeling of motion artifacts in contactless heart rate measurements"
    Tobias Wartzek et al.
    Computing in Cardiology 2013.
    """
    idx = 0
    while idx < len(signal):
        artifact_probability = 0.01
        if (scistat.bernoulli.rvs(artifact_probability)):
            exp_lambda = 2.8
            t_duration = scistat.expon.rvs(scale=1/exp_lambda)
            n_duration = min(int(t_duration*fs), len(signal)-idx)
            if (n_duration > 16):
                noise_len  = np.ceil(scift.next_fast_len(n_duration)/2)*2
                noise_len  = int(np.ceil(n_duration/2)*2)

                # Abitrary initial distribution that seems to produces the target pdf after filtering
                noise     = scistat.t.rvs(3, loc=0.0, scale=4, size=noise_len)
                f_range   = np.linspace(0, fs/2, num=int(np.floor(noise_len/2)))

                H_half    = np.ones(int(np.floor(noise_len/2)))
                H_half[f_range >= 1] = np.power(f_range[f_range >= 1], -1.4)

                H          = np.hstack([H_half, 0])
                F          = scift.rfft(noise)
                F_filt     = F*H
                noise_filt = np.real(scift.irfft(F_filt))
                signal[idx:idx+n_duration] += noise_filt[0:n_duration]
            idx += n_duration
        else:
            idx += 10
    return signal

def lowfrequency_noise(signal, fs):
    noise_probability = 1
    if (scistat.bernoulli.rvs(noise_probability)):
        exp_lambda = 0.2
        sigma      = scistat.expon.rvs(scale=1/exp_lambda)
        scale      = 3 + np.random.randn()*0.5

        n_duration = len(signal)
        noise_len  = n_duration
        noise      = np.random.randn(n_duration)*sigma
        f_range    = np.linspace(0, fs/2, num=int(np.floor(noise_len/2)))

        H_half    = np.ones(int(np.floor(noise_len/2)))
        H_half[f_range >= 1] = np.power(f_range[f_range >= 1], -scale)

        H          = np.hstack([H_half, 0])
        F          = scift.rfft(noise)
        F_filt     = F*H
        noise_filt = np.real(scift.irfft(F_filt))
        signal    += noise_filt
    return signal

def measurement_noise(signal):
    n          = len(signal)
    exp_lambda = 15
    scale      = scistat.expon.rvs(scale=1/exp_lambda)
    signal    += np.random.randn(n)*scale
    return signal

def sign_flipping(signal):
    flip_probability = 0.1
    if (scistat.bernoulli.rvs(flip_probability)):
        signal = -1*signal
    return signal

def truncation(signal, fs):
    truncation_probability = 0.5
    if (scistat.bernoulli.rvs(truncation_probability)):
        n_duration = len(signal)
        exp_mean   = n_duration/fs/4
        crop_time  = scistat.expon.rvs(scale=exp_mean)   
        crop_len   = int(min(np.round(crop_time*fs), n_duration*0.7))
        signal[-crop_len:] = 0
    return signal

def ecg_augmentation(signal):
    fs     = 500
    #signal = sign_flipping(signal)
    signal = lowfrequency_noise(signal, fs)
    signal = truncation(signal, fs)
    signal = motion_artifact(signal, fs)
    signal = measurement_noise(signal)
    return signal

def ecg_augmentation_partial(signal):
    fs     = 500
    #signal = sign_flipping(signal)
    signal = motion_artifact(signal, fs)
    signal = measurement_noise(signal)
    return signal

def sinus_arrhythmia(signal, fs):
    resp_f  = 0.25   
    n       = len(signal)
    T       = n*fs
    t       = np.linspace(0, T, num=n)
    signal += np.sin(2*np.pi*resp_f*t)
    return signal

# fs = 500
# T  = 10
# n  = round(fs*T)
# t  = np.linspace(0, T, num=n)
# f  = 0.5

# with open("../../all_leadall_train.pickle", "rb") as f:
#     data = pickle.load(f)    

# for datum in data:
#     signal        = np.array(datum, dtype=np.float64)
#     signal        = (signal - np.mean(signal)) / np.std(signal)
#     signal_origin = copy.deepcopy(signal)
#     signal_noisy  = ecg_augmentation(signal)
#     signal_noisy  = (signal_noisy - np.mean(signal_noisy)) / np.std(signal_noisy)
#     plt.figure(figsize=(10,1.5), tight_layout=True)
#     plt.plot(signal_origin)
#     plt.plot(signal_noisy)
#     plt.show()

# signal = motion_artifact(signal, fs)
# signal = measurement_noise(signal)
# plt.plot(signal_out)
# plt.plot(signal)
# plt.show()
