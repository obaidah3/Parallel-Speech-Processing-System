#ifndef FFT_UTILS_H
#define FFT_UTILS_H

#include <vector>
#include <complex>
#include <memory>
#include <unordered_map>
#include <mutex>
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

class FFTPlanPool {
public:
    static FFTPlanPool& instance() {
        thread_local FFTPlanPool pool;
        return pool;
    }

    FFTUtils& get(int fftSize) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = plans_.find(fftSize);
        if (it == plans_.end()) {
            auto [newIt, inserted] = plans_.emplace(fftSize, std::make_unique<FFTUtils>(fftSize));
            it = newIt;
        }
        return *it->second;
    }

    // Pre-allocate common sizes for optimization
    void preallocate(int fftSize) {
        get(fftSize);
    }

private:
    FFTPlanPool() = default;
    ~FFTPlanPool() = default;
    std::mutex mutex_;
    std::unordered_map<int, std::unique_ptr<FFTUtils>> plans_;
};

#endif