import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math


class attention_LDA:
   def __init__(self, concatenated_attention_arrays, concatenated_labels_arrays,
                freqs_scales, title, f_s):

      n_samples, n_features = concatenated_attention_arrays.shape
      classes = np.unique(concatenated_labels_arrays)
      n_classes = len(classes)

      # Calculate within and between class covariance
      mean = np.mean(concatenated_attention_arrays,axis=0)
      Sw = np.zeros((n_features,n_features))
      Sb = np.zeros((n_features,n_features))
      for c in classes:
         Xc = concatenated_attention_arrays[concatenated_labels_arrays==c]
         class_means = np.mean(Xc,axis=0)
         # Within-class variance
         Sw += (Xc-class_means).T.dot(Xc-class_means)
         mean_diff = (class_means-mean).reshape(n_features,1)
         # Between-class variance
         Sb += n_classes * (mean_diff).dot(mean_diff.T)

      show_covariance(Sb, title + '- Between Class Covariance', freqs_scales, f_s)
      show_covariance(Sw, title + '- Within Class Covariance', freqs_scales, f_s)
      show_covariance(np.linalg.inv(Sw), title + '- Inverse of Within Class Convariance', freqs_scales, f_s)

      # Find LDA covariance
      LDA_covarinace = np.linalg.inv(Sw).dot(Sb)
      eigen_values, eigen_vectors = np.linalg.eig(LDA_covarinace)
      eigen_vectors = eigen_vectors.T
      show_covariance(LDA_covarinace, title + '- LDA Covariance', freqs_scales, f_s)

      # Find eigen values of LDA covariance
      sorted_idxs = np.argsort(abs(eigen_values))[::-1]
      eigen_values,eigen_vectors = eigen_values[sorted_idxs],eigen_vectors[sorted_idxs]
      explained_variance_ratio = eigen_values / np.sum(eigen_values)
      abs_values = abs(explained_variance_ratio)
      print('abs_eigen_values', abs_values)

      # Create the bar chart of eigen values
      plt.figure(figsize=(10, 6))  # optional: set figure size
      sns.barplot(x=list(range(len(abs_values[:10]))), y=abs_values[:10], color='blue')
      ax = plt.gca()
      plt.xlabel('Index of Eigen Values', fontsize=18)  # Increase xlabel font size
      plt.ylabel('Abs(Eigen Values)', fontsize=18)  # Increase ylabel font size
      plt.title(title +'- Eigen Values of LDA Covariance', fontsize=17, fontweight='bold')  # Increase title font size
      ax.tick_params(axis='x', labelsize=17)
      ax.tick_params(axis='y', labelsize=17)
      plt.savefig('./'+title+'EigenValues.pdf', format='pdf')
      plt.show()

      # Focus on the first component
      n_components = 1
      major_linear_discriminants = eigen_vectors[0:n_components]
      show_eingen_vector([list(abs(major_linear_discriminants))[0]], freqs_scales, f_s)


def show_covariance(data, title, freqs_scales, f_s):
   # Here we show the covariance as the heatmap
   step = 20
   freq_ticks_indices = np.arange(0, len(freqs_scales), step)
   freq_ticks_values = [int(freqs_scales[i]) for i in freq_ticks_indices]
   freq_ticks_labels = [str(freq_ticks_values[i]) for i in range(len(freq_ticks_values))]
   plt.figure(figsize=(10, 8))  # Adjust the figure size if needed
   ax = sns.heatmap(data, cmap="ocean", yticklabels=False)
   ax.set_yticks(freq_ticks_indices)
   ax.set_yticklabels(freq_ticks_labels, fontsize=12)  # Increase y-tick label font size
   ax.set_xticks(freq_ticks_indices)
   ax.set_xticklabels(freq_ticks_labels, fontsize=12)  # Increase x-tick label font size
   plt.xlabel(f_s,  fontsize=17)  # Increase x-axis label font size
   plt.ylabel(f_s,  fontsize=17)  # Increase y-axis label font size
   plt.title(title,  fontsize=17, fontweight='bold')  # Increase title font size
   plt.savefig('./'+ title + '.pdf', format='pdf')
   plt.show()



def show_eingen_vector(data, freqs_scales, f_s):
   freqs = np.array(freqs_scales, dtype=float)
   data = np.array(data[0], dtype=float)
   # First we find the important frequencies based on LDA covariance (top 25% local maximums)
   # We have window size as 4, and we choose the member with the first highest eigen value in each window ==> 25%
   window = 4
   top_k = 1
   marked_indices = set()
   start = 0
   while start + window <= len(data):
      end = int(start + window)  # end is exclusive
      window_data = data[start:end]
      window_indices = np.argsort(window_data)[-top_k:]
      global_indices = [start + idx for idx in window_indices]
      marked_indices.update(global_indices)
      # Move to next window based on actual size used
      start = end  # NOT start += window
   print('marked_indices (top ratio)', len(marked_indices) / len(data), )
   marked_indices = sorted(list(marked_indices))
   top_freqs = freqs[marked_indices]
   top_values = data[marked_indices]
   top_freqs_rounded = [round(f, 2) for f in top_freqs]
   print('top_LDA_freqs_rounded', len(top_freqs_rounded)/len(freqs), top_freqs_rounded)


   # How to find these frequencies is discussed in the paper. They depend on the faulty type. Here they are harmonics of 7.73
   mechanical_freq = [7.73, 15.46, 23.19, 30.92, 38.65, 46.38, 54.11, 61.84,
                69.57, 77.3]
   # Here we let the mechanical frequencies to have +/-1 error
   mechanical_freq_used =[]
   for fr in mechanical_freq:
      mechanical_freq_used.append(math.ceil(fr))
      mechanical_freq_used.append(math.ceil(fr)-1)
      mechanical_freq_used.append(math.ceil(fr) + 1)

   # We try to identify the top 25% frequencies found by LDA in mechanical faulty frequencies
   top_freq_aligned = []
   top_values_aligned = []
   for i in range(len(top_freqs)):
      # 100 is the maximum frequency range
      if top_freqs[i]< 100:
         if top_freqs[i] in mechanical_freq_used:
            # The line below is to make sure we do not have repetitions in the aligned frequencies
            if (top_freqs[i]-1 not in top_freq_aligned and top_freqs[i] + 1 not in top_freq_aligned
                    and top_freqs[i] + 2 not in top_freq_aligned) and top_freqs[i] - 2 not in top_freq_aligned:
                  top_freq_aligned.append(top_freqs[i])
                  top_values_aligned.append(top_values[i])
   sns.set()
   plt.figure(figsize=(12, 6))
   plt.plot(freqs, data, marker='o', linestyle='-', markersize=3)
   # Red scatter points (Frequencies found by LDA that are in mechanical faulty frequencies)
   plt.scatter(top_freq_aligned, top_values_aligned, color='red', s=50, label='Local maximums aligned with mechanical knowledge')
   # Annotate each red point with its x and y values
   for x, y in zip(top_freq_aligned, top_values_aligned):
      plt.text(x, y, f'({x:.2f}, {y:.2f})', fontsize=12, color='black', ha='left', va='bottom')
   plt.xlabel(f_s, fontsize=18)
   plt.ylabel('Amplitude', fontsize=18)
   plt.title("Attention Data", fontsize=18, fontweight='bold')
   # Show the windows that are used to find local maximums
   for x in np.arange(0, max(freqs), window):
      if x == 0:
         plt.axvline(x=x, color='purple', linestyle='--', linewidth=0.7, alpha=0.7, label='Windows to find local maximum')
      else:
         plt.axvline(x=x, color='purple', linestyle='--', linewidth=0.7, alpha=0.7)

   plt.grid(True)
   plt.legend(fontsize=16)
   plt.tight_layout()
   plt.savefig('./'+ "Attention Data" + '.pdf', format='pdf')
   plt.show()

   print('aligned frequncies', top_freq_aligned)
