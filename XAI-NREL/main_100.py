# Import necessary libraries
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
import scipy.io
writer = SummaryWriter()
import copy
from datetime import datetime
import torch
import numpy as np
from einops import rearrange
import pickle
from torchinfo import summary
import random
import pandas as pd
from data_preprocessing.data_preprocessing import data_preprocessing
from torchvision import transforms


class main:
    def __init__(self, epochs, learning_rate, n_layers, n_head, use_CWT, freq0_scale1, wavelet_type, desired_scales_freqs_range, selected_time_division, window_size, signal):
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
        # Read faulty data
        faulty_signal = scipy.io.loadmat('./data/Damaged/D1.mat')
        df_faulty = pd.DataFrame(faulty_signal[signal], columns=[signal])
        df_faulty = pd.concat([df_faulty])

        # Read healthy data
        healthy_signal = scipy.io.loadmat('./data/Healthy/H1.mat')
        df_healthy = pd.DataFrame(healthy_signal[signal], columns=[signal])
        df_healthy = pd.concat([df_healthy])

        fault_data_pre = data_preprocessing(df_faulty, window_size, signal)
        fault_tensor = fault_data_pre.seq_tensor

        health_data_pre = data_preprocessing(df_healthy, window_size, signal)
        health_tensor = health_data_pre.seq_tensor

        data = torch.cat((health_tensor, fault_tensor), dim=0)
        data = data.to(device)
        print('data', data.shape)

        # We consider healthy and one faulty class
        num_classes = 2
        len_labels = [health_tensor.shape[0], fault_tensor.shape[0]]
        all_labels = []

        for i in range(num_classes):
            all_labels.append(torch.tensor([i] * len_labels[i]))
        # Concatenate the tensors in the all labels list
        labels = torch.cat(all_labels)
        labels = labels.to(device)

        log = 1
        from feature_extraction.wavelet_feature_extraction import wavelet_feature_extraction
        if use_CWT == 1:
            data = wavelet_feature_extraction(data, sampling_rate, wavelet_type, freq0_scale1, desired_scales_freqs_range, selected_time_division, log).data
        print('data_after_feature_extraction.shape', data.shape)

        # Normalize the data
        n_samples = data.shape[0]
        data = data.to(device)
        mean = torch.mean(data)
        std = torch.std(data)
        normalize = transforms.Normalize(mean, std)
        nor_data = normalize(data)
        nor_data = rearrange(nor_data, 'k i j ->k j i')
        num_features = nor_data.shape[1]
        nor_data = nor_data.to(device)

        # Shuffle both nor_data and labels using the shuffled indices
        shuffled_indices = list(range(n_samples))
        random.shuffle(shuffled_indices)
        shuffled_nor_data = nor_data[shuffled_indices]
        shuffled_labels = labels[shuffled_indices]

        # Create a TensorDataset with the shuffled nor_data and shuffled labels
        dataset = TensorDataset(shuffled_nor_data, shuffled_labels)

        # Divide into train, test and validation
        full_size = int(len(dataset))
        train_size = int(0.7 * full_size)
        val_size = (len(dataset) - train_size) // 2
        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size,
                                                                                           full_size - train_size - val_size],
                                                                                 generator=torch.Generator().manual_seed(
                                                                                     42))
        batch_size = 32
        # By setting shuffle=True in the DataLoader, you ensure that the data samples are shuffled before creating the batches.
        dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator())
        dataloader_val = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator())
        dataloader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, generator=torch.Generator())

        # Set model parameters
        # d_model is the window size: dmodel=num_features
        # seq_len is the number of frequencies
        dmodel = nor_data.shape[1]
        d_hid = 256  # Dimension of the feedforward network model in nn.TransformerEncoder
        n_layers = n_layers  # Number of nn.TransformerEncoderLayer in nn.TransformerEncoder
        n_head = n_head  # Number of heads in nn.MultiheadAttention
        sequence_length = nor_data.shape[-1]
        print('dmodelNum', dmodel)
        print('sequence_length', sequence_length)
        print('d_hid', d_hid)
        print('n_layers', n_layers)
        print('n_head', n_head)

        # Build the model with the parameters
        from prediction.model.transformer_model import my_transformer_model
        model = my_transformer_model(dmodel, sequence_length, n_head, d_hid, n_layers, num_classes, scale=16).to(device)
        print(summary(model))
        # Empty loss arrays for clean performance figures
        val_losses = []
        train_losses = []
        # CrossEntropyLoss, which is commonly used for classification tasks.
        loss_fn = nn.CrossEntropyLoss()
        # This line defines the optimizer to be used for updating the model parameters during training
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.0)
        # The variables are initialized
        loss_of_best_accuracy = float('inf')
        best_accuracy = 0
        best_model = None

        # Train loop and saving the best model
        start_time = datetime.now()
        from prediction.training import train_loop_func
        from prediction.validation import validation_loop_func
        for t in range(epochs):
            print(f"\nepoch {t + 1}\n-------------------------------")
            train_loop = train_loop_func(dataloader_train, model, loss_fn, optimizer, t + 1)
            train_losses.append(train_loop.train_loss)
            # Next, it calls the validation_loop function, passing the validation data, model, loss function, and the current epoch.
            # This function evaluates the model on the validation data, calculates the validation loss, and computes the accuracy.
            validation_loop = validation_loop_func(dataloader_val, model, loss_fn, t + 1)
            val_losses.append(validation_loop.validation_loss)
            accuracy = validation_loop.accuracy
            loss = validation_loop.validation_loss
            if accuracy > best_accuracy or (accuracy == best_accuracy and loss < loss_of_best_accuracy):
                best_accuracy = accuracy
                loss_of_best_accuracy = loss
                best_model = copy.deepcopy(model)
                print('model saved')
        end_time = datetime.now()
        print(f"\nDone! Validation accuracy: {(100 * best_accuracy):>0.2f}%")
        print(f"Training took {(end_time - start_time).total_seconds():>0.2f} seconds")


        # Predict the labels using the best model and store the attention data for all layers and heads
        # In the next lines, I only use all_attn_heads_layers, so you can ignore all_attn_heads_raw
        # attn_raw is the attention before softmax
        @torch.no_grad()
        def predict(model, loader):
            all_attn = torch.zeros((batch_size, sequence_length, num_features))
            all_attn_heads_layers = torch.zeros((n_head, batch_size, sequence_length, num_features))
            # Initialise empty tensors for predictions and targets
            all_preds_probs = torch.tensor([], device=device, dtype=int)
            all_preds = torch.tensor([], device=device, dtype=int)
            all_targets = torch.tensor([], device=device, dtype=int)
            model.eval()
            for batch in loader:
                data, labels = batch
                # This line performs the forward pass through the model to obtain the predicted class labels.
                preds, attn, attn_heads_layers, self.wq, self.wk, self.wv, attn_raw = model(
                    data.float())  # Assuming model returns two outputs
                attn = attn.to('cpu')
                layer_number = len(attn_heads_layers) - 1
                all_attn = torch.cat((all_attn, attn), dim=0)
                all_attn_heads_layers = torch.cat((all_attn_heads_layers, attn_heads_layers[layer_number].to('cpu')), dim=1)
                all_preds_probs = torch.cat((all_preds_probs, preds) , dim=0)
                preds = preds.argmax(1)
                # This line concatenates the preds with the existing all_preds tensor along the specified dimension (dim=0)
                all_preds = torch.cat((all_preds, preds) , dim=0)
                # Similarly, this line concatenates the target class labels (labels) with the existing all_targets tensor.
                all_targets = torch.cat((all_targets, labels.int()), dim=0)
            model.train()
            all_attn_heads_layers = rearrange(all_attn_heads_layers, 'head b l k -> b  k l head')
            return (all_preds_probs, all_preds, all_targets, all_attn[32:, :, :], all_attn_heads_layers[32:, :, :, :])

        # Predict train data labels, find the confusion matrix, and store the attention data regarding the train data
        from prediction.results import results
        with torch.no_grad():
            predictions_probs, predictions, targets, attn, attn_heads_layers = predict(best_model, dataloader_train)
            results_train_attention_heads = []
            for head in range(n_head):
                filtered_attn_heads_layers = rearrange(attn_heads_layers, 'b l k head -> b k l head')
                filtered_attn_heads_layers = filtered_attn_heads_layers[:, :, :, head]
                filtered_attn_heads_layers = torch.squeeze(filtered_attn_heads_layers, -1)
                results_train_attention_heads.append(
                    results(predictions_probs, predictions, targets, filtered_attn_heads_layers, num_classes))
            results_train = results(predictions_probs, predictions, targets, attn, num_classes)

        # Predict test data labels, find the confusion matrix
        with torch.no_grad():
            predictions_probs, predictions, targets, attn, attn_heads_layers = predict(best_model, dataloader_test)
        results_test = results(predictions_probs, predictions, targets, attn, num_classes)

        # Predict validation data labels, find the confusion matrix
        with torch.no_grad():
            predictions_probs, predictions, targets, attn, attn_heads_layers = predict(best_model, dataloader_val)
        results_val = results(predictions_probs, predictions, targets, attn, num_classes)

        # Filter attention data of a specific head
        attn_label_head = {}
        for i in range(num_classes):
            attn_label_head[i] = []

        for head in range(n_head):
            for i in range(num_classes):
                attn_label_head[i].append(results_train_attention_heads[head].attn_label_dict[str(i)])


        self.train_results = results_train.performance_results
        self.train_results = {"Train_" + key: value for key, value in self.train_results.items()}
        self.val_results = results_val.performance_results
        self.val_results = {"Validation_" + key: value for key, value in self.val_results.items()}
        self.test_results = results_test.performance_results
        self.test_results = {"Test_" + key: value for key, value in self.test_results.items()}
        self.result_dict = {'n_samples':[n_samples]}
        self.result_dict.update(self.train_results)
        self.result_dict.update(self.val_results)
        self.result_dict.update(self.test_results)

        # In case we have used CWT for feature extraction, the attention data is useful for interpretation and we store it.
        if use_CWT == 1 :
            for head in range(n_head):
                prefix = signal + '_attn_' + 'head' + str(head)
                attn_label = {}
                for i in range(num_classes):
                    attn_label[i] = attn_label_head[i][head]
                    with open('./explanation/' + prefix + 'L' + str(i)  + '_window_size' + str(window_size)+ '_freq0_scale1_'+str(freq0_scale1) +'_100.pickle', 'wb') as f:
                        pickle.dump(attn_label[i], f)


epochs = 200
learning_rate = 0.001
# N-Features (or window size) must be divisible by num_heads
# In our paper num_features = window size and sequence_length = number of frequencies
n_head = 1
n_layers = 2
# We can use either scales or frequencies in our analysis. There relation between scale and frequency is discussed in the paper
freq0_scale1 = 0
# Here the frequency or scale range is given.
desired_scales_freqs_range = np.arange(1, 100, 1)
desired_scales_freqs_range_string = f"np.arange({desired_scales_freqs_range[0]}, {desired_scales_freqs_range[-1]+desired_scales_freqs_range[1] - desired_scales_freqs_range[0]}, {desired_scales_freqs_range[1] - desired_scales_freqs_range[0]})"
# As an example, in a time window with length 500, we want to use every 8 member for the analysis not all the 500 time points.
selected_time_division = 8


df_result = pd.DataFrame()
# The name of the signal that we want to analyze
for signal in ['AN3', 'AN4', 'AN5', 'AN6', 'AN7', 'AN8', 'AN9', 'AN10']:
    # If we want to do feature extraction using CWT or not
    for use_CWT in [1, 0]:
        # The time window length for the wavelet analysis
        for window_size in [500, 50]:
            # The wavelet type
            for wavelet_type in ['gaus4']:
                results = main(epochs=epochs, learning_rate=learning_rate, n_layers=n_layers, n_head=n_head, use_CWT=use_CWT, freq0_scale1= freq0_scale1, wavelet_type=wavelet_type,
                            desired_scales_freqs_range=desired_scales_freqs_range, selected_time_division=selected_time_division, window_size = window_size, signal=signal).result_dict
                print('results', results)
                all_results = {'epochs':[epochs], 'learning_rate':[learning_rate], 'n_layers':[n_layers], 'n_head':[n_head], 'use_CWT':[use_CWT], 'freq0_scale1': [freq0_scale1],
                 'wavelet_type':[wavelet_type], 'desired_scales_freqs_range':[desired_scales_freqs_range_string], 'selected_time_division':[selected_time_division], 'window_size': [window_size], 'signal':[signal]}
                all_results.update(results)
                df_1 = pd.DataFrame(all_results)
                df_result = pd.concat([df_result, df_1], axis = 0)
                # Address to store the results
                df_result.to_csv('./'+"NREL_"+str(freq0_scale1)+'_'+"_100.csv", index=False, sep=';')

