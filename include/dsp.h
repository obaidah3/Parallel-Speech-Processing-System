#ifndef DSP_H
#define DSP_H

#include <complex>
#include <vector>

// Flat spectrogram for cache-efficient storage
class Spectrogram {
public:
    Spectrogram(int numFrames, int numBins);
    Spectrogram(const std::vector<std::vector<std::complex<float>>>& nested);

    int numFrames() const { return numFrames_; }
    int numBins() const { return numBins_; }

    std::complex<float>& operator()(int frame, int bin);
    const std::complex<float>& operator()(int frame, int bin) const;
    // Removed unsafe frame() methods that used reinterpret_cast

    void resize(int numFrames, int numBins);

private:
    int numFrames_;
    int numBins_;
    std::vector<std::complex<float>> data_;
};

// DSP preprocessing functions
void normalizeAudio(std::vector<float>& samples);
void removeDcOffset(std::vector<float>& samples);
void applyPreEmphasis(std::vector<float>& samples, float alpha);

// STFT / ISTFT with flat spectrogram
void computeSTFT(const std::vector<float>& samples, Spectrogram& spectrogram, int fftSize = 2048, int hopSize = 1024);
void computeISTFT(const Spectrogram& spectrogram, std::vector<float>& output, int fftSize = 2048, int hopSize = 1024);

// Filtering
void applySpeechFilter(std::vector<std::complex<float>>& frame, int sampleRate);

#endif