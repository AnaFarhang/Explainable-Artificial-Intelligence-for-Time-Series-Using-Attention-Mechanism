# Import necessary libraries
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
import numpy as np



class data_preprocessing:
    def __init__(self, data, seq_len, var):
        data_column=data[var].tolist()
        # Divide the signal into time windows: we need first to remove the remained elements
        elements_to_remove = len(data_column) % seq_len
        # Remove the remaining elements
        if elements_to_remove:
            data_column = data_column[:-elements_to_remove]
        data_seq= [data_column[i:i + seq_len] for i in range(0, len(data_column), seq_len)]

        try:
            seq_numeric = [[float(item) for item in sublist] for sublist in data_seq]
        except TypeError:
            seq_numeric = [[item for item in sublist] for sublist in data_seq]

        # Convert the list of lists to a NumPy array with float32 data type
        seq_array = np.array(seq_numeric)

        # Convert the reshaped NumPy array to a PyTorch tensor
        seq_tensor = torch.tensor(seq_array)
        # We return the divided signal as a tensor
        self.seq_tensor = torch.unsqueeze(seq_tensor, dim=1)


