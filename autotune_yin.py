#!/usr/bin/env python
# coding: utf-8

# In[239]:


import os
import torch
import torchaudio
from IPython.display import Audio
import numpy as np
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import random
from collections import Counter
from tqdm import tqdm
from matplotlib.pyplot import imshow


# In[229]:


wav_path = '/Users/ginoprasad/autotune/m4a_files/lalala.m4a'


# In[230]:


wav, sample_rate = torchaudio.load(wav_path)
wav = wav[0]
Audio(wav, rate=sample_rate)


# In[231]:


wav = wav[int(sample_rate*12.25):][:int(sample_rate*5.25)]


# In[ ]:


# wav = torch.Tensor(np.expand_dims(wav.numpy(), 0))
# path = "lalala.wav"
# torchaudio.save(
#     path, waveform, sample_rate,
#     encoding="PCM_S", bits_per_sample=16)
# inspect_file(path)


# In[232]:


Audio(wav, rate=sample_rate)


# # Smoothening

# In[3]:


# Importing Numpy package
import numpy as np
 
# sigma(standard deviation) and muu(mean) are the parameters of gaussian
 

def gaussian_function(kernel_size, sigma=1, muu=None):
    assert kernel_size % 2
    muu = kernel_size // 2

    # Initializing value of x,y as grid of kernel size
    # in the range of kernel size
 
    x = np.linspace(0, kernel_size-1, kernel_size)
    dst = np.sqrt(x**2)
 
    # lower normal part of gaussian
    normal = 1/(2 * np.pi * sigma**2)
 
    # Calculating Gaussian filter
    gauss = np.exp(-((dst-muu)**2 / (2.0 * sigma**2))) * normal
    return gauss
 
gaussian = gaussian_function(kernel_size=11, sigma=5) * 1.5
# print(gaussian)


# In[4]:


wav = scipy.ndimage.convolve1d(wav, gaussian)


# In[5]:


Audio(wav, rate=sample_rate)


# # Assigning Pitches to Frequencies

# In[284]:


prelim_notes_octave = 4
prelim_notes = [("C", 261.63), ("C#", 277.18), ("D", 293.66), ("D#", 311.13),
         ("E", 329.63), ("F", 349.23), ("F#", 369.99), ("G", 392.00), ("G#", 415.30), ("A", 440.00), ("A#", 466.16), ("B", 493.88), ]
# prelim_notes


# In[285]:


notes = [(freq * (2 ** (octave - prelim_notes_octave)), f"{name}{octave}") for octave in range(8) for name, freq in prelim_notes]
pd.DataFrame(notes)


# # Hyperparameters

# In[560]:


min_frequency = 60 # hertz
max_frequency = 120 # hertz
k = 0.15 # seconds
W = 0.05 # seconds
precision = 1000


# # Step 1: distance
# 
# Ideally: $x_t - x_{t+T} = 0$ for all t

# $d_t (\tau) = \sum_{j=t+1}^{t+W} (x_j - x_{j+\tau})^2$

# $= r_t(0) + r_{t+\tau}(0) - 2r_t(\tau)$

# Where $r_t(\tau) = \sum_{j=t+1}^{t+W} x_j x_{j+\tau}$

# In[561]:


def randints(n, k):
    return [random.randint(0, n-1) for _ in range(k)]


# In[562]:


def d(wav_slice, t):
    if type(wav_slice) != np.ndarray:
        wav_slice = wav_slice.numpy()
    autocorrelation = scipy.signal.convolve(wav_slice[t:], wav_slice[t:t+int(W*sample_rate)][::-1], mode='valid')
    energy = scipy.signal.convolve(wav_slice[t:] * wav_slice[t:], np.ones(int(W*sample_rate)), mode='valid')
    distance = (energy + energy[0]) - (2 * autocorrelation)
    assert len(distance) > 10
#     print(len(distance))
    return distance


# In[563]:


def normalized_d(wav_slice, t):
    sum_ = 0
    ret = []
    distance = d(wav_slice, t)
    for tau, dist in enumerate(distance):
        if tau == 0:
            ret.append(1)
        else:
            if sum_ == 0:
                ret.append(1)
            else:
                ret.append(dist / ((1/tau) * sum_))
        sum_ += dist
    return np.array(ret), distance


# In[564]:


def parabolic_interpolation(y):
    x = np.array(range(len(y)))
    x_squared = x ** 2
    ones = np.ones(len(y))

    mat = np.transpose(np.array([x_squared, x, ones]))
    if len(y) < 3:
        return np.argmin(y)
    a, b, c = np.matmul(np.linalg.inv(np.matmul(np.transpose(mat), mat)), np.transpose(mat)).dot(y)
    if a == 0 or -(b / (2 * a)) < 0:
        return np.argmin(y)
    return -(b / (2 * a))


# In[565]:


def pitch(wav_slice, x_min_threshold=sample_rate//max_frequency, x_max_threshold=sample_rate//min_frequency, y_threshold=0.15, t=0, width=3, index=False):
    ls, dist = normalized_d(wav_slice, t)
    minimum = None
    for x, val in enumerate(ls):
        if x_min_threshold <= x <= x_max_threshold and val < y_threshold and x < len(ls) - 1 and ls[x+1] >= val:
            minimum = x
            break
    if minimum is None:
        return None
    minimum = max(minimum-width, 0) + parabolic_interpolation(dist[max(minimum-width, 0):minimum+width+1]) # parabolic interpolation
    if index:
        return sample_rate / minimum, minimum
    return sample_rate / minimum


# In[566]:


def get_median_pitch(pitch_candidates):
    if type(pitch_candidates) != np.ndarray:
        pitch_candidates = np.array(pitch_candidates)
    closest = np.vectorize(lambda x: pd.DataFrame(notes)[1][np.argmin(pd.DataFrame(notes)[0] - x)])
    closest_notes = closest(pitch_candidates)
    mode = max(Counter(closest_notes).items(), key=lambda x: x[1])[0]
    return np.median(pitch_candidates[closest_notes == mode])    


# In[567]:


def pitch_predict(wav_slice, iterations=30):
    pitch_candidates = []
    for t in randints(int(k*sample_rate), iterations):
#         t=int(k*sample_rate) // 2
        pitch_ = pitch(wav_slice, t=t)
        if pitch_ is not None:
            pitch_candidates.append(pitch_)
    if not pitch_candidates:
        return None
    median = np.median(pitch_candidates)
    return median if median in pitch_candidates else pitch_candidates[0]


# In[568]:


def get_kmers(wav):
    return [wav[i:i+int(2*k*sample_rate)] for i in range(0, len(wav), int(k*sample_rate)) if i+int(2*k*sample_rate) <= len(wav)]


# In[569]:


amplitude = 1
def get_frequency(frequency, length):
    base = np.arange(0, length*sample_rate).astype(np.float64)
    c = (frequency * 2 * np.pi) / sample_rate
    wavelet_ = amplitude * np.sin(c * base)
    return wavelet_


# In[570]:


def notes_in_scale(root, major=True):
    pattern = [True, False, True, False, True, True, False, True, False, True, False, True]
    scale = pd.DataFrame(prelim_notes)[0]
    index = list(scale).index(root)
    if not major:
        index += 3
        index %= len(scale)
    scale = np.concatenate([scale[index:], scale[:index]])[pattern]
#     print(scale)
    return [x for x in notes if any(x[1][:-1] == y for y in scale)]


# In[571]:


def predict_scale(freqs):
    min_root, min_root_val = None, float('inf')
    for root, _ in prelim_notes:
        scale = notes_in_scale(root)
        scale_freqs = [freq for freq, _ in scale]
        closest = np.array([min((scale_freq for scale_freq in scale_freqs), key=lambda x: 1-abs(x/freq)) for freq in freqs])
        error = np.linalg.norm(frequencies - closest, ord=2)
        if error < min_root_val:
            min_root = root
            min_root_val = error
    return min_root


# In[592]:


def autotune(wav_slice, scale=None, precision=1000, max_scale=0.5, log=False, freq=None):
    if freq is None:
        freq = pitch_predict(wav_slice)
    closest_index = np.argmin(np.abs(pd.DataFrame(scale)[0] - freq)) if freq is not None else None
    closest = pd.DataFrame(scale)[0][closest_index] if closest_index is not None else None
    if freq is None or np.abs(1-(freq/closest)) > max_scale:
        print("NO CLOSEST\n---------")
        return wav_slice[:int(k * sample_rate)].numpy(), None

    if log:
        print(f"CLOSEST: {pd.DataFrame(scale)[1][closest_index]}")
        print(f"Scaling by {(freq/closest)}")
        print('--------')
    
    corrected = torchaudio.functional.resample(wav_slice[:int(k * sample_rate)], orig_freq=precision, new_freq=int(precision * freq/closest)).numpy()
    return corrected, closest


# In[593]:


# frequencies = []
# for kmer in tqdm(get_kmers(wav)):
#     frequencies.append(pitch_predict(kmer))
# frequencies = np.array(frequencies)


# In[594]:


# # root = predict_scale(frequencies)
# root = 'G#'
# scale = notes_in_scale(root, True)


# In[595]:


def process_kmer(kmer, closest,period_count=1):
    if closest is None:
        return kmer
    periods = kmer[len(kmer)//2:(len(kmer)//2) + (period_count*int(sample_rate / closest))]
    return periods


# In[596]:


# Audio(process_kmer(wav, closest = 98), rate=sample_rate)


# In[610]:


# amplitude = 1


# In[611]:


# prev_closest = None
# prev_processed = None

# autotuned = []
# for kmer, freq in zip(get_kmers(wav), frequencies):
#     auto_kmer, closest = autotune(kmer, freq=freq, scale=scale)
    
#     if closest is not None and (prev_closest and prev_closest == closest):
#         processed_kmer = prev_processed
#     elif closest is not None:
#         processed_kmer = process_kmer(auto_kmer, closest)
#         prev_processed = processed_kmer
#     else:
#         processed_kmer = None
#     prev_closest = closest

    
#     if processed_kmer is not None and closest == prev_closest:
#         processed_kmer_wav = np.concatenate([processed_kmer for _ in range((len(auto_kmer) // len(processed_kmer))+1)])
#         processed_kmer_wav = processed_kmer_wav[:len(auto_kmer)]
#         auto_kmer += amplitude * processed_kmer_wav
        
#     autotuned.append(auto_kmer)
# autotuned = np.concatenate(autotuned).astype(np.float64)


# In[612]:


# Audio(autotuned, rate=sample_rate)


# In[ ]:





# In[121]:


# Audio(autotuned + get_frequency(440, length=len(autotuned) / sample_rate) * 0.2, rate=sample_rate)


# In[47]:


# Audio(wav + get_frequency(440, length=len(wav) / sample_rate) * 0.2, rate=sample_rate)


# In[18]:


# len(get_frequency(closest, len(corrected) / sample_rate) * 0.1)


# In[694]:


# len(kmer)


# In[662]:


# root = 'G4'
# root = get_frequency(pd.DataFrame(notes)[0][list(pd.DataFrame(notes)[1]).index(root)], length=len(autotuned) / sample_rate)
# root = np.concatenate([root, np.zeros(len(autotuned) - len(root))]) * 0.01

# Audio(autotuned, rate=sample_rate)


# In[456]:


# root = 'G4'
# root = get_frequency(pd.DataFrame(notes)[0][list(pd.DataFrame(notes)[1]).index(root)], length=len(autotuned) / sample_rate)
# root = np.concatenate([root, np.zeros(len(wav) - len(root))]) * 0.05

# Audio(wav + root, rate=sample_rate)


# In[235]:


# def lengthen(wav_slice, duration=5):
#     if type(wav_slice) != np.ndarray:
#         wav_slice = wav_slice.numpy()
        
        
#     inserts = []
#     while len(wav_slice) / sample_rate < duration:
#         t = random.randint(0, int(len(wav_slice) - W - 1))
#         freq, index = pitch(wav_slice, t=t, index=True)

#         index = int(np.floor(index) + np.random.binomial(1, index % 1))
#         duration -= index / sample_rate
#         inserts.append((t, wav_slice[t:t+index]))
#     inserts = sorted(inserts, key=lambda x: -x[0])
    
#     for insert, wav_insert_slice in inserts:
#         wav_slice = np.concatenate([wav_slice[:t], wav_insert_slice, wav_slice[t:]])
# #     print(inserts)
    
# # #     new = 
# # #     while len()
# #     new = np.concatenate
# #     print(t)
        
# #     print(freq)
# #     pass
#     return wav_slice
# Audio(lengthen(wav[150000:250000]), rate=sample_rate)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




