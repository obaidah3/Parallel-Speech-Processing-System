#include "features.h"
#include <vector>
#include <complex>
#include <cmath>
#include <numeric>
#include <fstream>
#include <iostream>

// Calculate RMS energy
float calculateRMS(const std::vector<float>& samples) {
    float sum = 0.0f;
    for (float sample : samples) {
        sum += sample * sample;
    }
    return std::sqrt(sum / samples.size());
}

// Calculate Zero Crossing Rate
float calculateZCR(const std::vector<float>& samples) {
    int crossings = 0;
    for (size_t i = 1; i < samples.size(); ++i) {
        if ((samples[i-1] >= 0 && samples[i] < 0) || (samples[i-1] < 0 && samples[i] >= 0)) {
            crossings++;
        }
    }
    return static_cast<float>(crossings) / samples.size();
}

// Calculate spectral centroid
float calculateSpectralCentroid(const std::vector<std::complex<float>>& frame, int sampleRate) {
    float numerator = 0.0f;
    float denominator = 0.0f;
    int fftSize = (frame.size() - 1) * 2;

    for (size_t k = 0; k < frame.size(); ++k) {
        float magnitude = std::abs(frame[k]);
        float freq = k * sampleRate / static_cast<float>(fftSize);
        numerator += freq * magnitude;
        denominator += magnitude;
    }

    return denominator > 0 ? numerator / denominator : 0.0f;
}

// Calculate spectral bandwidth
float calculateSpectralBandwidth(const std::vector<std::complex<float>>& frame, int sampleRate, float centroid) {
    float numerator = 0.0f;
    float denominator = 0.0f;
    int fftSize = (frame.size() - 1) * 2;

    for (size_t k = 0; k < frame.size(); ++k) {
        float magnitude = std::abs(frame[k]);
        float freq = k * sampleRate / static_cast<float>(fftSize);
        numerator += (freq - centroid) * (freq - centroid) * magnitude;
        denominator += magnitude;
    }

    return denominator > 0 ? std::sqrt(numerator / denominator) : 0.0f;
}

// Calculate spectral flatness
float calculateSpectralFlatness(const std::vector<std::complex<float>>& frame) {
    float geometricMean = 1.0f;
    float arithmeticMean = 0.0f;
    int count = 0;

    for (const auto& bin : frame) {
        float magnitude = std::abs(bin);
        if (magnitude > 0) {
            geometricMean *= magnitude;
            arithmeticMean += magnitude;
            count++;
        }
    }

    if (count == 0) return 0.0f;

    geometricMean = std::pow(geometricMean, 1.0f / count);
    arithmeticMean /= count;

    return arithmeticMean > 0 ? geometricMean / arithmeticMean : 0.0f;
}

// Extract MFCC (simplified implementation)
std::vector<float> extractMFCC(const std::vector<std::complex<float>>& frame, int sampleRate, int numCoeffs) {
    // Simplified MFCC: just return magnitude spectrum as coefficients
    std::vector<float> mfcc(numCoeffs);
    for (int i = 0; i < numCoeffs && i < static_cast<int>(frame.size()); ++i) {
        mfcc[i] = std::abs(frame[i]);
    }
    return mfcc;
}

// Extract features from spectrogram and save to CSV
void extractFeatures(const std::vector<std::vector<std::complex<float>>>& spectrogram, int sampleRate, const std::string& outputFile) {
    std::ofstream file(outputFile);
    file << "Frame,RMS,ZCR,Centroid,Bandwidth,Flatness";

    for (int i = 0; i < 13; ++i) {
        file << ",MFCC" << i;
    }
    file << "\n";

    for (size_t frame = 0; frame < spectrogram.size(); ++frame) {
        const auto& specFrame = spectrogram[frame];

        // Convert complex to real for RMS and ZCR (simplified)
        std::vector<float> realFrame(specFrame.size());
        for (size_t i = 0; i < specFrame.size(); ++i) {
            realFrame[i] = specFrame[i].real();
        }

        float rms = calculateRMS(realFrame);
        float zcr = calculateZCR(realFrame);
        float centroid = calculateSpectralCentroid(specFrame, sampleRate);
        float bandwidth = calculateSpectralBandwidth(specFrame, sampleRate, centroid);
        float flatness = calculateSpectralFlatness(specFrame);
        auto mfcc = extractMFCC(specFrame, sampleRate, 13);

        file << frame << "," << rms << "," << zcr << "," << centroid << "," << bandwidth << "," << flatness;
        for (float coeff : mfcc) {
            file << "," << coeff;
        }
        file << "\n";
    }

    file.close();
}