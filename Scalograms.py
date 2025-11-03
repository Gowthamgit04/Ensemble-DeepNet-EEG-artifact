import numpy as np
import matplotlib.pyplot as plt
import os
import pywt  # Importing the PyWavelets library

def compute_cwt(data, scales, wavelet_name='cmor1.5-1.0'):
    # Perform a Continuous Wavelet Transform (CWT)
    coefficients, frequencies = pywt.cwt(data, scales, wavelet_name)
    return coefficients, frequencies

def plot_and_save_scalogram(coefficients, frequencies, path):
    # Generate a scalogram
    plt.figure(figsize=(10, 4))
    # Determine the intensity range based on the specific data
    magnitude = np.abs(coefficients)
    vmax = np.percentile(magnitude, 99)  # Using 99th percentile as max to avoid outliers
    vmin = magnitude.min()
    plt.imshow(magnitude, extent=[0, 1, frequencies.min(), frequencies.max()],
               cmap='viridis', aspect='auto', interpolation='nearest', origin='lower',
               vmin=vmin, vmax=vmax)  # Apply local scaling for intensity
    plt.colorbar(label='Magnitude')
    plt.xlabel('Time')
    plt.ylabel('Frequency (Hz)')
    plt.title('Scalogram')
    plt.savefig(path)
    plt.close()

def load_processed_files(log_path):
    if os.path.exists(log_path):
        with open(log_path, 'r') as file:
            return set(file.read().splitlines())
    return set()

def mark_file_as_processed(log_path, file_name):
    with open(log_path, 'a') as file:
        file.write(file_name + '\n')

def main():
    main_folder_path = '.\\Gowtham_test'
    output_folder_path = '.\\Scalograms_seg_artifacts'
    log_file_path = '.\\processed_files.txt'
    
    processed_files = load_processed_files(log_file_path)
    
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    
    for subfolder_name in os.listdir(main_folder_path):
        subfolder_path = os.path.join(main_folder_path, subfolder_name)
        output_subfolder_path = os.path.join(output_folder_path, subfolder_name)
        
        if not os.path.exists(output_subfolder_path):
            os.makedirs(output_subfolder_path)
        
        if os.path.isdir(subfolder_path):
            for file_name in os.listdir(subfolder_path):
                if file_name.endswith('.txt'):
                    file_path = os.path.join(subfolder_path, file_name)
                    output_file_path = os.path.join(output_subfolder_path, f"{os.path.splitext(file_name)[0]}_scalogram.png")
                    
                    if file_name in processed_files:
                        print(f"Skipping already processed file: {file_name}")
                        continue
                    
                    try:
                        eeg_data = np.loadtxt(file_path)
                        scales = np.arange(1, 128)  # Define scales
                        cwt_coeffs, frequencies = compute_cwt(eeg_data, scales)
                        plot_and_save_scalogram(cwt_coeffs, frequencies, output_file_path)
                        mark_file_as_processed(log_file_path, file_name)
                        print(f"Processed and saved: {file_name}")
                    except Exception as e:
                        print(f"Error processing {file_name}: {e}")

if __name__ == '__main__':
    main()
