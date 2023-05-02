#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


wav_path = '/Users/ginoprasad/autotune/m4a_files/lalala.m4a'


# In[3]:


wav, sample_rate = torchaudio.load(wav_path)
wav = wav[0]
wav = wav[int(sample_rate*12.25):][:int(sample_rate*5.25)]
Audio(wav, rate=sample_rate)


# # Assigning Pitches to Frequencies

# In[4]:


prelim_notes_octave = 4
prelim_notes = [("C", 261.63), ("C#", 277.18), ("D", 293.66), ("D#", 311.13),
         ("E", 329.63), ("F", 349.23), ("F#", 369.99), ("G", 392.00), ("G#", 415.30), ("A", 440.00), ("A#", 466.16), ("B", 493.88), ]
# prelim_notes


# In[5]:


notes = [(freq * (2 ** (octave - prelim_notes_octave)), f"{name}{octave}") for octave in range(8) for name, freq in prelim_notes]
pd.DataFrame(notes)


# # Hyperparameters

# In[6]:


min_frequency = 60 # hertz
max_frequency = 120 # hertz
W = 0.05 # seconds
precision = 1000


# # Step 1: distance
# 
# Ideally: $x_t - x_{t+T} = 0$ for all t

# $d_t (\tau) = \sum_{j=t+1}^{t+W} (x_j - x_{j+\tau})^2$

# $= r_t(0) + r_{t+\tau}(0) - 2r_t(\tau)$

# Where $r_t(\tau) = \sum_{j=t+1}^{t+W} x_j x_{j+\tau}$

# In[7]:


def randints(n, k):
    return [random.randint(0, n-1) for _ in range(k)]


# In[8]:


def d(wav_slice, t):
    if type(wav_slice) != np.ndarray:
        wav_slice = wav_slice.numpy()
    autocorrelation = scipy.signal.convolve(wav_slice[t:], wav_slice[t:t+int(W*sample_rate)][::-1], mode='valid')
    energy = scipy.signal.convolve(wav_slice[t:] * wav_slice[t:], np.ones(int(W*sample_rate)), mode='valid')
    distance = (energy + energy[0]) - (2 * autocorrelation)
    assert len(distance) > 10
#     print(len(distance))
    return distance


# In[9]:


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


# In[10]:


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


# In[27]:


def pitch(wav_slice, y_threshold=0.15, t=0, width=3, index=False):
    ls, dist = normalized_d(wav_slice, t)
    minimum = None
    for x, val in enumerate(ls):
        if x and min_frequency <= sample_rate / x <= max_frequency and val < y_threshold and x < len(ls) - 1 and ls[x+1] >= val:
            minimum = x
            break
    if minimum is None:
        return None
    minimum = max(minimum-width, 0) + parabolic_interpolation(dist[max(minimum-width, 0):minimum+width+1]) # parabolic interpolation
    if index:
        return sample_rate / minimum, minimum
    return sample_rate / minimum


# In[28]:


def get_median_pitch(pitch_candidates):
    if type(pitch_candidates) != np.ndarray:
        pitch_candidates = np.array(pitch_candidates)
    closest = np.vectorize(lambda x: pd.DataFrame(notes)[1][np.argmin(pd.DataFrame(notes)[0] - x)])
    closest_notes = closest(pitch_candidates)
    mode = max(Counter(closest_notes).items(), key=lambda x: x[1])[0]
    return np.median(pitch_candidates[closest_notes == mode])    


# In[29]:


def pitch_predict(wav_slice, iterations=30):
    pitch_candidates = []
    for t in randints(len(wav_slice)-int(W + np.ceil(sample_rate / max_frequency)), iterations):
        pitch_ = pitch(wav_slice, t=t)
        if pitch_ is not None:
            pitch_candidates.append(pitch_)
    if not pitch_candidates:
        return None
    median = np.median(pitch_candidates)
    return median if median in pitch_candidates else pitch_candidates[0]


# In[30]:


amplitude = 1
def get_frequency(frequency, length):
    base = np.arange(0, length*sample_rate).astype(np.float64)
    c = (frequency * 2 * np.pi) / sample_rate
    wavelet_ = amplitude * np.sin(c * base)
    return wavelet_


# In[31]:


pitch_predict(wav)


# In[ ]:




