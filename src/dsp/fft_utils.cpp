#include "fft_utils.h"
#include <cmath>
#include <cstring>

FFTUtils::FFTUtils(int fftSize) : fftSize_(fftSize) {
    inputBuffer_ = fftwf_alloc_real(fftSize);
    outputBuffer_ = fftwf_alloc_complex(fftSize / 2 + 1);

    // Create plans
    forwardPlan_ = fftwf_plan_dft_r2c_1d(fftSize, inputBuffer_, outputBuffer_, FFTW_MEASURE);
    inversePlan_ = fftwf_plan_dft_c2r_1d(fftSize, outputBuffer_, inputBuffer_, FFTW_MEASURE);
}

FFTUtils::~FFTUtils() {
    fftwf_destroy_plan(forwardPlan_);
    fftwf_destroy_plan(inversePlan_);
    fftwf_free(inputBuffer_);
    fftwf_free(outputBuffer_);
}

void FFTUtils::forward(const std::vector<float>& input, std::vector<std::complex<float>>& output) {
    // Copy input to buffer
    std::memcpy(inputBuffer_, input.data(), fftSize_ * sizeof(float));

    // Execute FFT
    fftwf_execute(forwardPlan_);

    // Copy output
    output.resize(fftSize_ / 2 + 1);
    for (int i = 0; i < fftSize_ / 2 + 1; ++i) {
        output[i] = std::complex<float>(outputBuffer_[i][0], outputBuffer_[i][1]);
    }
}

void FFTUtils::inverse(const std::vector<std::complex<float>>& input, std::vector<float>& output) {
    // Copy input to buffer
    for (int i = 0; i < fftSize_ / 2 + 1; ++i) {
        outputBuffer_[i][0] = input[i].real();
        outputBuffer_[i][1] = input[i].imag();
    }

    // Execute IFFT
    fftwf_execute(inversePlan_);

    // Copy output and normalize
    output.resize(fftSize_);
    for (int i = 0; i < fftSize_; ++i) {
        output[i] = inputBuffer_[i] / fftSize_;
    }
}

std::vector<float> FFTUtils::generateHannWindow(int size) {
    std::vector<float> window(size);
    for (int i = 0; i < size; ++i) {
        window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (size - 1)));
    }
    return window;
}