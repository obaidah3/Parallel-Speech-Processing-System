#include "dsp.h"
#include "fft_utils.h"
#include <vector>
#include <complex>
#include <algorithm>

// Compute inverse STFT using overlap-add
void computeISTFT(const Spectrogram& spectrogram, std::vector<float>& output, int fftSize, int hopSize) {
    FFTUtils& fft = FFTPlanPool::instance().get(fftSize);
    std::vector<float> window = FFTUtils::generateHannWindow(fftSize);

    int numFrames = spectrogram.numFrames();
    int outputSize = (numFrames - 1) * hopSize + fftSize;
    std::vector<float> overlapBuffer(outputSize, 0.0f);
    std::vector<float> windowSum(outputSize, 0.0f);

    for (int frame = 0; frame < numFrames; ++frame) {
        // Prepare frame data
        std::vector<std::complex<float>> frameSpec(fftSize / 2 + 1);
        for (int bin = 0; bin < fftSize / 2 + 1; ++bin) {
            frameSpec[bin] = spectrogram(frame, bin);
        }

        // Inverse FFT
        std::vector<float> frameData;
        fft.inverse(frameSpec, frameData);

        // Apply window
        for (int i = 0; i < fftSize; ++i) {
            frameData[i] *= window[i];
        }

        // Overlap-add
        int start = frame * hopSize;
        for (int i = 0; i < fftSize; ++i) {
            overlapBuffer[start + i] += frameData[i];
            windowSum[start + i] += window[i] * window[i];
        }
    }

    // Normalize by window sum
    output.resize(outputSize);
    for (int i = 0; i < outputSize; ++i) {
        if (windowSum[i] > 0.0f) {
            output[i] = overlapBuffer[i] / windowSum[i];
        } else {
            output[i] = overlapBuffer[i];
        }
    }
}