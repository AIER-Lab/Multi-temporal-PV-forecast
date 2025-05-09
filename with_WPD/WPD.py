import numpy as np
import pandas as pd
import pywt
from pywt import WaveletPacket
import os

# -------- INPUTS --------
file_path = '' 
target_col = ''
datetime_col = ''
output_path = ''
wavelet = 'db8'
maxlevel = 2

# -------- LOAD DATA --------
df = pd.read_csv(file_path, parse_dates=[datetime_col])
original_signal = df[target_col].values
datetime_series = df[datetime_col]
original_length = len(original_signal)

# -------- DECOMPOSE AND RECONSTRUCT --------
wp = WaveletPacket(data=original_signal, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)
sub_signals = {}

for node in wp.get_level(maxlevel, 'natural'):
    path = node.path
    wp_temp = WaveletPacket(data=np.zeros(original_length), wavelet=wavelet, mode='symmetric')
    wp_temp[path] = node.data
    rec = wp_temp.reconstruct(update=False)

    rec = rec[:original_length] if len(rec) > original_length else np.pad(rec, (0, original_length - len(rec)), 'constant')
    sub_signals[path] = rec

# -------- SAVE TO SINGLE CSV --------
result_df = pd.DataFrame({datetime_col: datetime_series})
for path, signal in sub_signals.items():
    result_df[f'SS_{path}'] = signal

result_df.to_csv(output_path, index=False)
