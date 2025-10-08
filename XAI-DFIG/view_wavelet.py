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
    def __init__(self, window_size, signal, wavelet_type, freq0_scale1, REU, speed, title):
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
        sampling_rate = 10000

        if signal == 'Current':
            var = 'Is'
        if signal == 'Power':
            var ='Pe'
        # Read and concatenate faulty signals
        faulty_signal1 = scipy.io.loadmat('./data/'+signal+'/'+REU+'REU/'+speed+'rpm/test1.mat')
        signal_array = faulty_signal1[var];
        df_faulty1 = pd.DataFrame(signal_array, columns=[var])
        faulty_signal2 = scipy.io.loadmat('./data/'+signal+'/'+REU+'REU/'+speed+'rpm/test2.mat')
        signal_array = faulty_signal2[var];
        df_faulty2 = pd.DataFrame(signal_array, columns=[var])
        faulty_signal3 = scipy.io.loadmat('./data/'+signal+'/'+REU+'REU/'+speed+'rpm/test3.mat')
        signal_array = faulty_signal3[var];
        df_faulty3 = pd.DataFrame(signal_array, columns=[var])
        faulty_signal4 = scipy.io.loadmat('./data/'+signal+'/'+REU+'REU/'+speed+'rpm/test4.mat')
        signal_array = faulty_signal4[var];
        df_faulty4 = pd.DataFrame(signal_array, columns=[var])
        faulty_signal5 = scipy.io.loadmat('./data/'+signal+'/'+REU+'REU/'+speed+'rpm/test5.mat')
        signal_array = faulty_signal5[var];
        df_faulty5 = pd.DataFrame(signal_array, columns=[var])
        df_faulty = pd.concat([df_faulty1, df_faulty2, df_faulty3, df_faulty4, df_faulty5])
        print('df_faulty', df_faulty.size)

        # Read and concatenate healthy signals
        healthy_signal1 = scipy.io.loadmat('./data/'+signal+'/healthy/'+speed+'rpm/test1.mat')
        signal_array = healthy_signal1[var];
        df_healthy1 = pd.DataFrame(signal_array, columns=[var])
        healthy_signal2 = scipy.io.loadmat('./data/'+signal+'/healthy/'+speed+'rpm/test2.mat')
        signal_array = healthy_signal2[var];
        df_healthy2 = pd.DataFrame(signal_array, columns=[var])
        healthy_signal3 = scipy.io.loadmat('./data/'+signal+'/healthy/'+speed+'rpm/test3.mat')
        signal_array = healthy_signal3[var];
        df_healthy3 = pd.DataFrame(signal_array, columns=[var])
        healthy_signal4 = scipy.io.loadmat('./data/'+signal+'/healthy/'+speed+'rpm/test4.mat')
        signal_array = healthy_signal4[var];
        df_healthy4 = pd.DataFrame(signal_array, columns=[var])
        healthy_signal5 = scipy.io.loadmat('./data/'+signal+'/healthy/'+speed+'rpm/test5.mat')
        signal_array = healthy_signal5[var];
        df_healthy5 = pd.DataFrame(signal_array, columns=[var])
        df_healthy = pd.concat([df_healthy1, df_healthy2, df_healthy3, df_healthy4, df_healthy5])
        print('df_healthy', df_healthy.size)

        # Divide the data into time windows
        fault_data_pre = data_preprocessing(df_faulty, window_size, var)
        fault_tensor = fault_data_pre.seq_tensor

        health_data_pre = data_preprocessing(df_healthy, window_size, var)
        health_tensor = health_data_pre.seq_tensor

        # Here we need to set max frequency
        maxfreq = 450
        # Show the wavelet transformation
        tenth_member_health = health_tensor.mean(dim=0)
        #tenth_member_health = health_tensor1[500, :, :]
        output_list_health = tenth_member_health.squeeze().tolist()
        view_wavelet_transform(torch.tensor(output_list_health).squeeze(), wavelet_type,
                               sampling_rate, maxfreq, freq0_scale1, window_size, 'Healthy samples_'+title)

        tenth_member_fault = fault_tensor.mean(dim=0)
        #tenth_member_fault = fault_tensor7[500, :, :]
        output_list_fault = tenth_member_fault.squeeze().tolist()
        view_wavelet_transform(torch.tensor(output_list_fault).squeeze(), wavelet_type,
                               sampling_rate, maxfreq, freq0_scale1, window_size, 'Faulty samples_'+title)



freq0_scale1 = 0
# The name of the signal that we want to analyze
for signal in ['Current', 'Power']:
    # If we want to do feature extraction using CWT or not
    for use_CWT in [1]:
        # The time window length for the wavelet analysis
        for window_size in [500]:
            # The unbalance state in the dataset
            for REU in ['300', '225', '150']:
                # The speed in the dataset
                for speed in ['1590', '1560', '1530']:
                    # The wavelet type
                    for wavelet_type in ['gaus4']:
                        view_wavelet(window_size = window_size, signal=signal, wavelet_type=wavelet_type, freq0_scale1 = freq0_scale1,
                                     REU = REU, speed = speed, title=signal+'_'+str(window_size)+'_'+REU+'_'+speed)
