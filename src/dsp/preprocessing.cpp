#include "audio.h"
#include <numeric>
#include <algorithm>
#include <cmath>

// Normalize audio samples to [-1, 1] range
// Finds the maximum absolute value and scales all samples accordingly
// If max abs is 0, does nothing (silent audio)
void normalizeAudio(std::vector<float>& samples) {
    if (samples.empty()) return;

    // Find the maximum absolute value
    float maxAbs = 0.0f;
    for (float sample : samples) {
        maxAbs = std::max(maxAbs, std::fabs(sample));
    }

    // Normalize if maxAbs > 0
    if (maxAbs > 0.0f) {
        for (float& sample : samples) {
            sample /= maxAbs;
        }
    }
}

// Remove DC offset by subtracting the mean
// Calculates the average of all samples and subtracts it from each sample
void removeDcOffset(std::vector<float>& samples) {
    if (samples.empty()) return;

    // Calculate mean using std::accumulate
    float mean = std::accumulate(samples.begin(), samples.end(), 0.0f) / samples.size();

    // Subtract mean from each sample
    for (float& sample : samples) {
        sample -= mean;
    }
}

// Apply pre-emphasis filter: y[n] = x[n] - alpha * x[n-1]
// This boosts high frequencies, useful for speech processing
// Processes in-place, modifying the input vector
void applyPreEmphasis(std::vector<float>& samples, float alpha) {
    if (samples.size() < 2) return;

    // Apply filter from the end to avoid overwriting previous values
    for (size_t n = samples.size() - 1; n > 0; --n) {
        samples[n] = samples[n] - alpha * samples[n - 1];
    }
    // Note: y[0] remains x[0] as there's no x[-1]
}