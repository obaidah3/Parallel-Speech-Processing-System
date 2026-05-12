#include "parallel.h"
#include "audio.h"
#include "dsp.h"
#include "fft_utils.h"
#include "audio_features.h"
#include "timing.h"
#include <omp.h>
#include <iostream>

// OpenMP parallel STFT
void ompStft(Spectrogram& spectrogram, const std::vector<float>& samples, int fftSize, int hopSize) {
    std::vector<float> window = FFTUtils::generateHannWindow(fftSize);

    int numFrames = (samples.size() - fftSize) / hopSize + 1;
    spectrogram.resize(numFrames, fftSize / 2 + 1);

    #pragma omp parallel
    {
        // Thread-local buffers to avoid allocation overhead
        std::vector<float> frameData(fftSize);
        std::vector<std::complex<float>> frameSpec(fftSize / 2 + 1);

        #pragma omp for schedule(dynamic)
        for (int frame = 0; frame < numFrames; ++frame) {
            int start = frame * hopSize;

            // Zero the frame buffer
            std::fill(frameData.begin(), frameData.end(), 0.0f);

            // Copy samples to frame
            int copySize = std::min(fftSize, static_cast<int>(samples.size() - start));
            std::copy(samples.begin() + start, samples.begin() + start + copySize, frameData.begin());

            // Apply window
            for (int i = 0; i < fftSize; ++i) {
                frameData[i] *= window[i];
            }

            // Compute FFT using a thread-local plan
            FFTUtils& fft = FFTPlanPool::instance().get(fftSize);
            fft.forward(frameData, frameSpec);
            for (int bin = 0; bin < fftSize / 2 + 1; ++bin) {
                spectrogram(frame, bin) = frameSpec[bin];
            }

            if (frame == 0) {
                #pragma omp critical
                std::cout << "OpenMP: Thread " << omp_get_thread_num() << " processing frame " << frame << std::endl;
            }
        }
    }
}

// OpenMP parallel filtering
void ompFiltering(Spectrogram& spectrogram, int sampleRate) {
    #pragma omp parallel for
    for (int frame = 0; frame < spectrogram.numFrames(); ++frame) {
        std::vector<std::complex<float>> frameData(spectrogram.numBins());
        for (int bin = 0; bin < spectrogram.numBins(); ++bin) {
            frameData[bin] = spectrogram(frame, bin);
        }
        applySpeechFilter(frameData, sampleRate);
        for (int bin = 0; bin < spectrogram.numBins(); ++bin) {
            spectrogram(frame, bin) = frameData[bin];
        }
    }
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
    Spectrogram spectrogram(0, 0);
    ompStft(spectrogram, samples, 2048, 1024);

    // Filtering with OpenMP
    ompFiltering(spectrogram, sampleRate);

    // Feature extraction
    extractFeatures(spectrogram, sampleRate, featuresFile);

    // ISTFT
    std::vector<float> reconstructed;
    computeISTFT(spectrogram, reconstructed);

    // Write output
    writeWavMono(outputFile, reconstructed, sampleRate);

    timer.stop();
    std::cout << "OpenMP processing time: " << timer.elapsedSeconds() << " seconds" << std::endl;
}