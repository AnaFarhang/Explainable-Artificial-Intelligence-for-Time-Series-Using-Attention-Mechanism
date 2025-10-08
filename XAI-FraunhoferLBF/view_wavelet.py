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
    def __init__(self, window_size, freq0_scale1, signal, fault_type, title):
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

        # The sampling_dict provides the recorded signals with their sampling rates.
        sampling_dict = {74000: ['brng_f_x', 'brng_f_y', 'brng_f_z', 'brng_r_x', 'brng_r_y', 'brng_r_z'],
                         37000: ['Nacl_x', 'Nacl_y', 'Nacl_z'],
                         2950: ['top_l_x', 'top_l_y', 'top_l_z', 'top_r_x', 'top_r_y', 'top_r_z',
                                'bot_f_x', 'bot_f_y', 'bot_f_z', 'bot_r_x', 'bot_r_y', 'bot_r_z']}

        for rate, variables in sampling_dict.items():
            if signal in variables:
                sampling_rate = rate

        # Read the data
        faulty_signal = scipy.io.loadmat('./data/Bearing/'+fault_type+'.mat')
        df_faulty = pd.DataFrame(faulty_signal[signal], columns=[signal])

        healthy_signal = scipy.io.loadmat('./data/Healthy/Healthy/Healthy_2023_11_03_100653.mat')
        df_healthy = pd.DataFrame(healthy_signal[signal], columns=[signal])

        # The data is big. Here, we reduce the data samples for faster computation
        df_faulty = df_faulty.iloc[:500 * 4000]
        df_healthy = df_healthy.iloc[:500 * 4000]

        # Divide the data into time windows
        fault_data_pre = data_preprocessing(df_faulty, window_size, signal)
        fault_tensor = fault_data_pre.seq_tensor
        health_data_pre = data_preprocessing(df_healthy, window_size, signal)
        health_tensor = health_data_pre.seq_tensor

        # Here we need to set max frequency
        maxfreq = 200
        # Show the wavelet transformation
        tenth_member_health = health_tensor.mean(dim=0)
        #tenth_member_health = health_tensor[500, :, :]
        output_list_health = tenth_member_health.squeeze().tolist()
        view_wavelet_transform(torch.tensor(output_list_health).squeeze(), wavelet_type, sampling_rate, maxfreq, freq0_scale1, window_size, 'Healthy samples_'+title)

        tenth_member_fault = fault_tensor.mean(dim=0)
        #tenth_member_fault = fault_tensor[500, :, :]
        output_list_fault = tenth_member_fault.squeeze().tolist()
        view_wavelet_transform(torch.tensor(output_list_fault).squeeze(), wavelet_type, sampling_rate, maxfreq, freq0_scale1, window_size, 'Faulty samples_'+title)


# In this dataset, there are different types of faults. Therefore, we specify them.
fault_type = 'OutterRace/OutterRace_2023_12_20_183828'
freq0_scale1 = 0
# The name of the signal that we want to analyze
for signal in ['brng_f_x']:
    # If we want to do feature extraction using CWT or not
    for use_CWT in [1]:
        # The time window length for the wavelet analysis
        for window_size in [500]:
            # The wavelet type
            for wavelet_type in ['gaus4']:
                view_wavelet(window_size = window_size, freq0_scale1 = freq0_scale1, signal=signal, fault_type = fault_type, title=signal+'_'+str(window_size))

