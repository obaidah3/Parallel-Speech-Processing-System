#ifndef AUDIO_H
#define AUDIO_H

#include <vector>
#include <string>
#include <utility>

// Struct for mono audio data
// Contains the audio samples as floating-point values and the sample rate
struct AudioData {
    std::vector<float> samples;  // Audio samples in float format
    int sampleRate;             // Sample rate in Hz
};

// WAV I/O functions for mono audio
// Reads a mono WAV file and returns a pair of (samples vector, sample rate)
// Throws runtime_error on failure or if not mono
std::pair<std::vector<float>, int> readWavMono(const std::string& filename);

// Writes a mono WAV file from samples and sample rate
// Throws runtime_error on failure
void writeWavMono(const std::string& filename, const std::vector<float>& samples, int sampleRate);

// DSP preprocessing functions
// Normalizes audio samples to the range [-1, 1]
void normalizeAudio(std::vector<float>& samples);

// Removes DC offset by subtracting the mean value
void removeDcOffset(std::vector<float>& samples);

// Applies pre-emphasis filter: y[n] = x[n] - alpha * x[n-1]
// Alpha typically 0.97 for speech processing
void applyPreEmphasis(std::vector<float>& samples, float alpha = 0.97f);

#endif