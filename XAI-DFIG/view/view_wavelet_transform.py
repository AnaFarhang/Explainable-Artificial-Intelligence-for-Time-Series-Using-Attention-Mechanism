import numpy as np
import matplotlib.pyplot as plt
import pywt
import seaborn as sns



class view_wavelet_transform:
  def __init__(self, data, wavelet_type, sampling_rate, max_freq, freq0_scale1, window_size, title):

      if freq0_scale1 ==0:
          desired_freqs_range= np.arange(1, max_freq, 1)
          f0 = pywt.scale2frequency(wavelet_type, 1)  # central frequency
          desired_scales = (f0 * sampling_rate) / (desired_freqs_range)
          desired_scales = list(dict.fromkeys(desired_scales))

      if freq0_scale1==1:
          # Specify the range of scales to be used (This needs adaptation based on the used scale)
          desired_scales = np.arange(round(sampling_rate/max_freq), round(sampling_rate/3), 40).tolist()
          desired_scales.reverse()

      sampling_period=1/sampling_rate
      X_numpy = data.cpu().detach().numpy()
      coeffs, freqs = pywt.cwt(X_numpy, desired_scales, wavelet_type, sampling_period=sampling_period)

      coeffs_magnitudes = np.abs(coeffs)
      log_coeffs_magnitudes = np.log(coeffs_magnitudes)

      # We have chosen step 50 for both time and frequency axes
      step = 50
      desired_freqs = np.arange(1, max_freq, 1)
      freq_ticks_indices = np.arange(0, len(desired_freqs), step)
      freq_ticks_values = [int(desired_freqs[i]) for i in freq_ticks_indices]
      freq_ticks_labels = [str(freq_ticks_values[i]) for i in range(len(freq_ticks_values))]
      plt.figure(figsize=(10, 8))
      ax = sns.heatmap(log_coeffs_magnitudes, cmap='ocean', yticklabels=False)
      ax.set_yticks(freq_ticks_indices)
      ax.set_yticklabels(freq_ticks_labels, fontsize=12)

      desired_times = np.arange(1, window_size, 1)
      time_ticks_indices = np.arange(0, len(desired_times), step)
      time_ticks_values = [int(desired_times[i]) for i in time_ticks_indices]
      time_ticks_labels = [str(time_ticks_values[i]) for i in range(len(time_ticks_values))]
      ax.set_xticks(time_ticks_indices)
      ax.set_xticklabels(time_ticks_labels, fontsize=13)
      plt.xlabel('Time (S)', fontsize=13)
      plt.ylabel('Frequencies (Hz)', fontsize=13)
      plt.title(title+'_ Log(abs(Wavelet Coefficients))', fontsize=13, fontweight='bold')
      plt.savefig('./view/DFIG_Train_Heatmap_Log(abs(Wavelet Coefficients))_'+title+'.pdf', format='pdf')
      plt.show()

