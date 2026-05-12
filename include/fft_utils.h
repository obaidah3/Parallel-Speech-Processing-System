#ifndef FFT_UTILS_H
#define FFT_UTILS_H

#include <vector>
#include <complex>
#include <fftw3.h>

// FFT utilities using FFTW3
class FFTUtils {
public:
    FFTUtils(int fftSize);
    ~FFTUtils();

    // Forward FFT
    void forward(const std::vector<float>& input, std::vector<std::complex<float>>& output);

    // Inverse FFT
    void inverse(const std::vector<std::complex<float>>& input, std::vector<float>& output);

    // Hann window generation
    static std::vector<float> generateHannWindow(int size);

private:
    int fftSize_;
    fftwf_plan forwardPlan_;
    fftwf_plan inversePlan_;
    float* inputBuffer_;
    fftwf_complex* outputBuffer_;
};

#endif