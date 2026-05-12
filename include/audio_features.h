#ifndef FEATURES_H
#define FEATURES_H

#include <vector>
#include <complex>
#include <string>
#include "dsp.h"

// Feature extraction functions
float calculateRMS(const std::vector<float>& samples);
float calculateZCR(const std::vector<float>& samples);
float calculateSpectralCentroid(const std::vector<std::complex<float>>& frame, int sampleRate);
float calculateSpectralBandwidth(const std::vector<std::complex<float>>& frame, int sampleRate, float centroid);
float calculateSpectralFlatness(const std::vector<std::complex<float>>& frame);
std::vector<float> extractMFCC(const std::vector<std::complex<float>>& frame, int sampleRate, int numCoeffs = 13);
std::vector<std::vector<float>> createMelFilterbank(int numFilters, int fftSize, int sampleRate);

// Batch feature extraction
void extractFeatures(const Spectrogram& spectrogram, int sampleRate, const std::string& outputFile);

#endif