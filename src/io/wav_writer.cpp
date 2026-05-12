#include "audio.h"
#include <sndfile.h>
#include <stdexcept>

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