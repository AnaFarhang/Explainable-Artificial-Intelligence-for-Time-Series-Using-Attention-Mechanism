import pickle
import torch
from einops import rearrange
import numpy as np

# Parameter setting
num_classes=2
heads = [0]
freq0_scale1 = 0
log = 1
wavelet = 'gaus4'
desired_scales_freqs_range = np.arange(1, 450, 1)
signal = 'Current'
REU = '300'
speed  = '1590'
window_size = 500

# Relation between scale and frequency is discussed in the paper
if freq0_scale1 == 1:
    freqs_scales = desired_scales_freqs_range
    f_s = "Scale (1/Hz)"
if freq0_scale1 == 0:
    desired_scales_freqs = list(dict.fromkeys(desired_scales_freqs_range))
    freqs_scales = desired_scales_freqs
    f_s = "Frequency (Hz)"

# Read attention data for all layers and heads
all_heads = {}
for i in range(num_classes):
    all_heads['L'+str(i)]={}

for head in heads:
    for i in range(num_classes):
        prefix = signal + REU + 'REU_' + speed + 'rpm' + '_attn_' + 'head' + str(head)
        with open('./' + prefix + 'L' + str(i)  + '_window_size' + str(window_size)+ '_freq0_scale1_'+str(freq0_scale1) +'_450.pickle', "rb") as f:
            loaded_tensor = pickle.load(f)
            # Calculate average over the time dimension, because we want to focus on frequency analysis
            mean_tensor_dim2 = torch.mean(loaded_tensor, dim=2)
            # Do this for all heads
            all_heads['L'+str(i)][str(head)] = mean_tensor_dim2


processed_attention_list =[]
labels_list = []
for i in range(num_classes):
    # Calculate average of attention data for all heads
    all_tensor_heads = []
    for head in heads:
        all_tensor_heads.append(all_heads['L'+str(i)][str(head)])
    stacked_tensors = torch.stack(all_tensor_heads)
    sum_tensors = torch.sum(stacked_tensors, dim=0)
    average_tensor = sum_tensors / len(heads)
    # We normalize the resulted tensor
    mean = torch.mean(average_tensor)
    std = torch.std(average_tensor)
    nor_average_tensor = (average_tensor - mean) / std
    processed_attention_array = nor_average_tensor.numpy()
    # Since our loop is about labels, we store the data of each iteration in a list
    processed_attention_list.append(processed_attention_array)
    labels_list.append(i*np.ones(nor_average_tensor.shape[0]))


concatenated_attention_arrays = np.concatenate(processed_attention_list, axis=0)
concatenated_labels_arrays = np.concatenate(labels_list, axis=0)

# Now we calculate LDA for the processed attention and label data
from attention_LDA import attention_LDA
attention_LDA(concatenated_attention_arrays, concatenated_labels_arrays, freqs_scales, 'Attention Data ', f_s)