import torch
import numpy as np
import matplotlib.pyplot as plt
import pywt
from matplotlib.colors import Normalize


class wavelet_feature_extraction:

  def __init__(self, data, sampling_rate, wavelet_type, freq0_scale1, desired_scales_freqs_range, selected_time_division, log):
      device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
      data= data.squeeze(dim=1)
      # We can work either with scale or frequency. But we need to keep in mind that the input of pywt.cwt is always scales
      # So if we have chosen frequency, we need to convert it to scales
      if freq0_scale1 ==0:
          f0 = pywt.scale2frequency(wavelet_type, 1)  # central frequency
          desired_scales = (f0 * sampling_rate) / (desired_scales_freqs_range)
          scales = list(dict.fromkeys(desired_scales))
      if freq0_scale1 ==1:
          scales = desired_scales_freqs_range
          #scales.reverse()


      tList = [compute_cwt(m, wavelet_type, sampling_rate, scales, selected_time_division, log, device)
               for i, m in enumerate(torch.unbind(data, dim=0))]

      self.data = torch.stack(tList)



def compute_cwt(X, wavelet, sampling_rate, scales, division, log, device):
      X_numpy = X.cpu().detach().numpy()
      # We find the wavelet coefficients
      coeffs, freqs = pywt.cwt(X_numpy, scales, wavelet, sampling_period=1/sampling_rate)
      coeffs = np.abs(coeffs)
      if log ==1:
          coeffs = np.log(np.where(coeffs <= 0, 1e-8, coeffs))
      # We do not use all the members, we sample every 8th member due to the performance efficiency
      coeffs2 = coeffs[:, ::division][:, :]
      return torch.tensor(coeffs2).to(device)
