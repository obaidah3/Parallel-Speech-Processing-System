#ifndef FEATURES_H
#define FEATURES_H

#include <vector>
#include <complex>
#include <string>

// Feature extraction functions
float calculateRMS(const std::vector<float>& samples);
float calculateZCR(const std::vector<float>& samples);
float calculateSpectralCentroid(const std::vector<std::complex<float>>& frame, int sampleRate);
float calculateSpectralBandwidth(const std::vector<std::complex<float>>& frame, int sampleRate, float centroid);
float calculateSpectralFlatness(const std::vector<std::complex<float>>& frame);
std::vector<float> extractMFCC(const std::vector<std::complex<float>>& frame, int sampleRate, int numCoeffs = 13);

// Batch feature extraction
void extractFeatures(const std::vector<std::vector<std::complex<float>>>& spectrogram, int sampleRate, const std::string& outputFile);

#endif