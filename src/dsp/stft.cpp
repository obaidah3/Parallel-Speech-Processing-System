#include "fft_utils.h"
#include <vector>
#include <complex>
#include <algorithm>

// Compute STFT of audio samples
// Uses Hann window, 50% overlap
void computeSTFT(const std::vector<float>& samples, std::vector<std::vector<std::complex<float>>>& spectrogram, int fftSize = 2048, int hopSize = 1024) {
    FFTUtils fft(fftSize);
    std::vector<float> window = FFTUtils::generateHannWindow(fftSize);

    int numFrames = (samples.size() - fftSize) / hopSize + 1;
    spectrogram.resize(numFrames, std::vector<std::complex<float>>(fftSize / 2 + 1));

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
    }
}