#!/usr/bin/env python
# coding: utf-8

# In[4]:


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
import numpy as np


# In[6]:


import yin_pitch_prediction as yin


# In[7]:


def get_hann_window(N):
    window = np.arange(N) * (2 * np.pi / N)
    return 0.5 * (1 - np.cos(window))


# In[8]:


def wrap(x):
    x = (x % (2 * np.pi))
    if x > np.pi:
        return x - (2 * np.pi)
    return x


# # Presets

# In[9]:


W = 1024
H = W // 8
hann_window = get_hann_window(W)


# # Loading Sample

# In[10]:


wav_path = '/Users/ginoprasad/autotune/m4a_files/lalala.m4a'


# In[13]:


sample, sample_rate = torchaudio.load(wav_path)
sample = sample[0].numpy()
sample = sample[int(sample_rate*12.25):][:int(sample_rate*5.25)]


# ### Padding Sample

# In[14]:


sample = np.concatenate([sample, np.zeros(W - (len(sample) % W))])
assert not len(sample) % W


# In[15]:


Audio(sample, rate=sample_rate)


# # Processing Splice

# In[16]:


def pitch_shift(splice, R, prev_analysis_phase, prev_synthesis_phase):
    # Squared Windowing, then FFT
    fft_transformed = fft(splice * hann_window)
    fft_synthesized = np.zeros_like(fft_transformed)
    
    synthesis_frequencies = np.zeros((W//2)+1)
    synthesis_amplitudes = np.zeros_like(synthesis_frequencies)
    for k, val in enumerate(fft_transformed[:(W//2)+1]):
        k_prime = int((R * k) + 0.5)
        if k_prime > W//2:
            continue

        angle, amplitude = np.angle(val), np.absolute(val)
        phase_difference = angle - prev_analysis_phase[k]
        
        central_frequency = (2 * np.pi * k) / W
        phase_deviation = wrap((phase_difference) - (central_frequency*H))
        frequency = ((W * phase_deviation) / (2 * np.pi * H)) + k
        
        synthesis_frequencies[k_prime] = frequency * R
        synthesis_amplitudes[k_prime] += amplitude
        
    for k, prev_phase in enumerate(prev_synthesis_phase):
        # Calculating appropriate phase
        phase = wrap(prev_phase + ((H * 2 * np.pi * synthesis_frequencies[k])/W))
        fft_synthesized[k] = synthesis_amplitudes[k] * (np.cos(phase) + (1j * np.sin(phase)))
    fft_synthesized[:W//2:-1] = np.conjugate(fft_synthesized[1:W//2])
    return np.real(ifft(fft_synthesized)), np.angle(fft_transformed[:(W//2)+1]), np.angle(fft_synthesized[:(W//2)+1])


# In[17]:


def robotization(splice):
    fft_transformed = fft(splice * hann_window[:len(splice)])
    zerod_phase = np.absolute(fft_transformed)
    return np.real(ifft(zerod_phase))


# # Hann Window

# In[23]:


def main(R):
    prev_analysis = np.zeros((W//2)+1)
    prev_shifted = np.zeros_like(prev_analysis)
    output = np.zeros_like(sample)
    for start_index in tqdm(range(0, len(sample)-W, H)):
        splice = sample[start_index:start_index+W]
        if len(splice) < W:
            break
        processed_splice, prev_analysis, prev_shifted = pitch_shift(splice, R, prev_analysis, prev_shifted)
    #     processed_splice = robotization(splice)
        output[start_index:start_index+len(splice)] += (hann_window[:len(splice)] * processed_splice)
    return output


# In[37]:


# Audio(main(1.5), rate=sample_rate, autoplay=True)


# In[ ]:





# In[ ]:




