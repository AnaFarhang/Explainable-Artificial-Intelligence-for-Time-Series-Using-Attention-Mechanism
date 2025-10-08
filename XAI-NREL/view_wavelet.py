# Import necessary libraries
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from torch.utils.tensorboard import SummaryWriter
import scipy.io
writer = SummaryWriter()
import torch
import numpy as np
import pandas as pd
from data_preprocessing.data_preprocessing import data_preprocessing
from view.view_wavelet_transform import view_wavelet_transform


class view_wavelet:
    def __init__(self, window_size, wavelet_type, freq0_scale1, signal, title):
        # This line assigns the value 42 to the variable RANDOM_SEED for reproducibility
        RANDOM_SEED = 42
        # This line sets the random seed for the PyTorch library for various purposes, such as weight initialization and shuffling.
        torch.manual_seed(RANDOM_SEED)
        # This line sets the random seed for the NumPy library.
        np.random.seed(RANDOM_SEED)
        # This line sets the random seed for all available CUDA devices, e.g., multiple GPUs.
        torch.cuda.manual_seed_all(RANDOM_SEED)
        # By doing so, it ensures that the results of convolution operations performed by cuDNN (CUDA Deep Neural Network)
        # are consistent across different runs given the same input and configuration.
        torch.backends.cudnn.deterministic = True
        # Disabling cuDNN benchmarking ensures consistent performance across different runs of your code.
        # By this command, the same input will always produce the same output, regardless of the hardware or runtime environment.
        torch.backends.cudnn.benchmark = False


        # If GPU device is available we use it.
        # This line is also in transformermodel class. Please change it to the wishing GPU
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print('Using {} device'.format(device))

        sampling_rate = 40000

        # Read the data
        faulty_signal = scipy.io.loadmat('./data/Damaged/D1.mat')
        df_faulty = pd.DataFrame(faulty_signal[signal], columns=[signal])
        df_faulty = pd.concat([df_faulty])

        healthy_signal = scipy.io.loadmat('./data/Healthy/H1.mat')
        df_healthy = pd.DataFrame(healthy_signal[signal], columns=[signal])
        df_healthy = pd.concat([df_healthy])

        # Divide the data into time windows
        fault_data_pre = data_preprocessing(df_faulty, window_size, signal)
        fault_tensor = fault_data_pre.seq_tensor

        health_data_pre = data_preprocessing(df_healthy, window_size, signal)
        health_tensor = health_data_pre.seq_tensor

        # Here we need to set max frequency
        maxfreq = 100
        # Show the wavelet transformation
        tenth_member_health = health_tensor.mean(dim=0)
        #tenth_member_health = health_tensor[500, :, :]
        output_list_health = tenth_member_health.squeeze().tolist()
        view_wavelet_transform(torch.tensor(output_list_health).squeeze(), wavelet_type, sampling_rate, maxfreq, freq0_scale1, window_size, 'Healthy samples_'+title)

        tenth_member_fault = fault_tensor.mean(dim=0)
        #tenth_member_fault = fault_tensor[500, :, :]
        output_list_fault = tenth_member_fault.squeeze().tolist()
        view_wavelet_transform(torch.tensor(output_list_fault).squeeze(), wavelet_type, sampling_rate, maxfreq, freq0_scale1, window_size, 'Faulty samples_'+title)


freq0_scale1=0
# The name of the signal that we want to analyze
for signal in ['AN3']:
    # If we want to do feature extraction using CWT or not
    for use_CWT in [1]:
        # The time window length for the wavelet analysis
        for window_size in [500]:
            # The wavelet type
            for wavelet_type in ['gaus4']:
                view_wavelet(window_size = window_size, wavelet_type=wavelet_type, freq0_scale1 = freq0_scale1, signal=signal, title=signal+'_'+str(window_size))

