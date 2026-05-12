#include <vector>
#include <complex>
#include <cmath>

// Apply low-pass filter in frequency domain
void applyLowPassFilter(std::vector<std::complex<float>>& frame, int sampleRate, float cutoffFreq) {
    int fftSize = (frame.size() - 1) * 2;
    for (size_t k = 0; k < frame.size(); ++k) {
        float freq = k * sampleRate / static_cast<float>(fftSize);
        if (freq > cutoffFreq) {
            frame[k] *= 0.0f;  // Attenuate
        }
    }
}

// Apply high-pass filter in frequency domain
void applyHighPassFilter(std::vector<std::complex<float>>& frame, int sampleRate, float cutoffFreq) {
    int fftSize = (frame.size() - 1) * 2;
    for (size_t k = 0; k < frame.size(); ++k) {
        float freq = k * sampleRate / static_cast<float>(fftSize);
        if (freq < cutoffFreq) {
            frame[k] *= 0.0f;  // Attenuate
        }
    }
}

// Apply band-pass filter in frequency domain
void applyBandPassFilter(std::vector<std::complex<float>>& frame, int sampleRate, float lowCutoff, float highCutoff) {
    int fftSize = (frame.size() - 1) * 2;
    for (size_t k = 0; k < frame.size(); ++k) {
        float freq = k * sampleRate / static_cast<float>(fftSize);
        if (freq < lowCutoff || freq > highCutoff) {
            frame[k] *= 0.0f;  // Attenuate
        }
    }
}

// Apply speech band-pass filter (300Hz - 3400Hz)
void applySpeechFilter(std::vector<std::complex<float>>& frame, int sampleRate) {
    applyBandPassFilter(frame, sampleRate, 300.0f, 3400.0f);
}

// Apply noise gate (simple threshold)
void applyNoiseGate(std::vector<std::complex<float>>& frame, float threshold) {
    for (auto& bin : frame) {
        if (std::abs(bin) < threshold) {
            bin *= 0.0f;
        }
    }
}