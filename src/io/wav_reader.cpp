#include "audio.h"
#include <sndfile.h>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <algorithm>

// Read mono WAV file and return samples and sample rate
// Uses libsndfile to read PCM WAV files
// Ensures the file is mono (channels == 1)
// Converts samples to float format
std::pair<std::vector<float>, int> readWavMono(const std::string& filename) {
    SF_INFO sfinfo;
    // Initialize sfinfo to zero
    memset(&sfinfo, 0, sizeof(sfinfo));

    // Open the file for reading
    SNDFILE* file = sf_open(filename.c_str(), SFM_READ, &sfinfo);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    // Check if the file is mono
    if (sfinfo.channels != 1) {
        sf_close(file);
        throw std::runtime_error("Only mono WAV files are supported: " + filename);
    }

    // Prepare vector for samples
    std::vector<float> samples(sfinfo.frames);

    // Read all frames into the vector
    sf_count_t read = sf_read_float(file, samples.data(), sfinfo.frames);
    if (read != sfinfo.frames) {
        sf_close(file);
        throw std::runtime_error("Failed to read all samples from: " + filename);
    }

    // Close the file
    sf_close(file);

    // Return samples and sample rate
    return {samples, sfinfo.samplerate};
}