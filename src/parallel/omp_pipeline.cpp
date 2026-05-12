#include "parallel.h"
#include "audio.h"
#include "dsp.h"
#include "fft_utils.h"
#include "features.h"
#include "timing.h"
#include <omp.h>
#include <iostream>

// OpenMP parallel STFT
void ompStft(std::vector<std::vector<std::complex<float>>>& spectrogram, const std::vector<float>& samples, int fftSize, int hopSize) {
    FFTUtils fft(fftSize);
    std::vector<float> window = FFTUtils::generateHannWindow(fftSize);

    int numFrames = (samples.size() - fftSize) / hopSize + 1;
    spectrogram.resize(numFrames, std::vector<std::complex<float>>(fftSize / 2 + 1));

    #pragma omp parallel for
    for (int frame = 0; frame < numFrames; ++frame) {
        int start = frame * hopSize;
        std::vector<float> frameData(fftSize, 0.0f);

        // Copy samples to frame
        int copySize = std::min(fftSize, static_cast<int>(samples.size() - start));
        std::copy(samples.begin() + start, samples.begin() + start + copySize, frameData.begin());

        // Apply window
        for (int i = 0; i < fftSize; ++i) {
            frameData[i] *= window[i];
        }

        // Compute FFT
        fft.forward(frameData, spectrogram[frame]);

        if (frame == 0) {
            #pragma omp critical
            std::cout << "OpenMP: Thread " << omp_get_thread_num() << " processing frame " << frame << std::endl;
        }
    }
}

// OpenMP parallel filtering
void ompFiltering(std::vector<std::vector<std::complex<float>>>& spectrogram, int sampleRate) {
    #pragma omp parallel for
    for (size_t frame = 0; frame < spectrogram.size(); ++frame) {
        applySpeechFilter(spectrogram[frame], sampleRate);
    }
}

// OpenMP parallel feature extraction
void ompFeatureExtraction(const std::vector<std::vector<std::complex<float>>>& spectrogram, int sampleRate, const std::string& outputFile) {
    extractFeatures(spectrogram, sampleRate, outputFile);
}

// Full OpenMP-only pipeline (no MPI)
void runOmpPipeline(const std::string& inputFile, const std::string& outputFile, const std::string& featuresFile) {
    Timer timer;
    timer.start();

    // Read audio
    auto [samples, sampleRate] = readWavMono(inputFile);
    std::cout << "OpenMP: Read " << samples.size() << " samples at " << sampleRate << " Hz" << std::endl;

    // Preprocessing
    normalizeAudio(samples);
    removeDcOffset(samples);
    applyPreEmphasis(samples);

    // STFT with OpenMP
    std::vector<std::vector<std::complex<float>>> spectrogram;
    ompStft(spectrogram, samples, 2048, 1024);

    // Filtering with OpenMP
    ompFiltering(spectrogram, sampleRate);

    // Feature extraction with OpenMP
    ompFeatureExtraction(spectrogram, sampleRate, featuresFile);

    // ISTFT
    std::vector<float> reconstructed;
    computeISTFT(spectrogram, reconstructed);

    // Write output
    writeWavMono(outputFile, reconstructed, sampleRate);

    timer.stop();
    std::cout << "OpenMP processing time: " << timer.elapsedSeconds() << " seconds" << std::endl;
}