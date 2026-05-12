#include "dsp.h"
#include "fft_utils.h"
#include <vector>
#include <complex>
#include <algorithm>

// Spectrogram implementation
Spectrogram::Spectrogram(int numFrames, int numBins) : numFrames_(numFrames), numBins_(numBins), data_(numFrames * numBins) {}

Spectrogram::Spectrogram(const std::vector<std::vector<std::complex<float>>>& nested) {
    numFrames_ = nested.size();
    numBins_ = nested.empty() ? 0 : nested[0].size();
    data_.reserve(numFrames_ * numBins_);
    for (const auto& frame : nested) {
        data_.insert(data_.end(), frame.begin(), frame.end());
    }
}

std::complex<float>& Spectrogram::operator()(int frame, int bin) {
    if (frame < 0 || frame >= numFrames_ || bin < 0 || bin >= numBins_) {
        throw std::out_of_range("Spectrogram index out of range");
    }
    return data_[frame * numBins_ + bin];
}

const std::complex<float>& Spectrogram::operator()(int frame, int bin) const {
    if (frame < 0 || frame >= numFrames_ || bin < 0 || bin >= numBins_) {
        throw std::out_of_range("Spectrogram index out of range");
    }
    return data_[frame * numBins_ + bin];
}

// Remove incorrect frame() methods - they use dangerous reinterpret_cast
// std::vector<std::complex<float>>& Spectrogram::frame(int frame) {
//     return *reinterpret_cast<std::vector<std::complex<float>>*>(&data_[frame * numBins_]);
// }

// const std::vector<std::complex<float>>& Spectrogram::frame(int frame) const {
//     return *reinterpret_cast<const std::vector<std::complex<float>>*>(&data_[frame * numBins_]);
// }

void Spectrogram::resize(int numFrames, int numBins) {
    numFrames_ = numFrames;
    numBins_ = numBins;
    data_.resize(numFrames * numBins);
}

// Compute STFT of audio samples
// Uses Hann window, 50% overlap
void computeSTFT(const std::vector<float>& samples, Spectrogram& spectrogram, int fftSize, int hopSize) {
    FFTUtils& fft = FFTPlanPool::instance().get(fftSize);
    std::vector<float> window = FFTUtils::generateHannWindow(fftSize);

    int numFrames = (samples.size() - fftSize) / hopSize + 1;
    spectrogram.resize(numFrames, fftSize / 2 + 1);

    // Pre-allocate frame buffer to avoid per-frame allocations
    std::vector<float> frameData(fftSize);
    std::vector<std::complex<float>> frameSpec(fftSize / 2 + 1);

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

        // Compute FFT
        fft.forward(frameData, frameSpec);
        for (int bin = 0; bin < fftSize / 2 + 1; ++bin) {
            spectrogram(frame, bin) = frameSpec[bin];
        }
    }
}