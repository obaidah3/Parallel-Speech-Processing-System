#include "audio.h"
#include <sndfile.h>
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

// Write mono WAV file
// Uses libsndfile to write PCM WAV files
// Assumes samples are in float format
void writeWavMono(const std::string& filename, const std::vector<float>& samples, int sampleRate) {
    SF_INFO sfinfo;
    // Set up sfinfo for mono WAV
    sfinfo.samplerate = sampleRate;
    sfinfo.channels = 1;
    sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;

    // Open the file for writing
    SNDFILE* file = sf_open(filename.c_str(), SFM_WRITE, &sfinfo);
    if (!file) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    // Write all samples to the file
    sf_count_t written = sf_write_float(file, samples.data(), samples.size());
    if (written != static_cast<sf_count_t>(samples.size())) {
        sf_close(file);
        throw std::runtime_error("Failed to write all samples to: " + filename);
    }

    // Close the file
    sf_close(file);
}