#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import scipy.signal

def plot_waveform(filename, title):
    sample_rate, data = wavfile.read(filename)
    
    # Handle stereo audio by taking first channel
    if len(data.shape) > 1:
        data = data[:, 0]
    
    # Convert to float for better plotting
    if data.dtype != np.float32:
        data = data.astype(np.float32)
        if data.max() > 1.0:  # Assume integer format
            data = data / 32768.0  # Normalize 16-bit audio
    
    time = np.arange(len(data)) / sample_rate
    
    plt.figure(figsize=(12, 4))
    plt.plot(time, data)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(title)
    plt.grid(True)
    plt.savefig(f'data/results/{title.lower().replace(" ", "_")}_waveform.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_spectrogram(filename, title):
    sample_rate, data = wavfile.read(filename)
    
    # Handle stereo audio by taking first channel
    if len(data.shape) > 1:
        data = data[:, 0]
    
    # Convert to float
    if data.dtype != np.float32:
        data = data.astype(np.float32)
        if data.max() > 1.0:  # Assume integer format
            data = data / 32768.0  # Normalize 16-bit audio
    
    # Adjust spectrogram parameters based on data length
    nperseg = min(2048, len(data) // 4)  # Use smaller window if data is short
    noverlap = nperseg // 2
    
    frequencies, times, Sxx = scipy.signal.spectrogram(data, fs=sample_rate, 
                                                     nperseg=nperseg, 
                                                     noverlap=noverlap)
    
    plt.figure(figsize=(12, 6))
    plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud', cmap='viridis')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title(title)
    plt.colorbar(label='Intensity [dB]')
    plt.savefig(f'data/results/{title.lower().replace(" ", "_")}_spectrogram.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    # Plot original and processed audio
    plot_waveform('data/raw/audio_1.wav', 'Original Audio Waveform')
    plot_spectrogram('data/raw/audio_1.wav', 'Original Audio Spectrogram')
    
    # Check if processed files exist
    import os
    if os.path.exists('data/output_sequential.wav'):
        plot_waveform('data/output_sequential.wav', 'Sequential Processed Audio Waveform')
        plot_spectrogram('data/output_sequential.wav', 'Sequential Processed Audio Spectrogram')
    
    if os.path.exists('data/output_omp.wav'):
        plot_waveform('data/output_omp.wav', 'OpenMP Processed Audio Waveform')
        plot_spectrogram('data/output_omp.wav', 'OpenMP Processed Audio Spectrogram')
    
    if os.path.exists('data/output_mpi.wav'):
        plot_waveform('data/output_mpi.wav', 'MPI Processed Audio Waveform')
        plot_spectrogram('data/output_mpi.wav', 'MPI Processed Audio Spectrogram')
    
    if os.path.exists('data/output_hybrid.wav'):
        plot_waveform('data/output_hybrid.wav', 'Hybrid Processed Audio Waveform')
        plot_spectrogram('data/output_hybrid.wav', 'Hybrid Processed Audio Spectrogram')

if __name__ == "__main__":
    main()