#include "audio.h"
#include <sndfile.h>
#include <stdexcept>
#include <iostream>

AudioData readWav(const std::string& filename) {
    SF_INFO sfinfo;
    SNDFILE* file = sf_open(filename.c_str(), SFM_READ, &sfinfo);
    if (!file) {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    AudioData audio;
    audio.sampleRate = sfinfo.samplerate;
    audio.channels = sfinfo.channels;
    audio.samples.resize(sfinfo.frames * sfinfo.channels);

    sf_count_t read = sf_read_float(file, audio.samples.data(), audio.samples.size());
    if (read != static_cast<sf_count_t>(audio.samples.size())) {
        sf_close(file);
        throw std::runtime_error("Failed to read all samples from: " + filename);
    }

    sf_close(file);
    return audio;
}

void writeWav(const std::string& filename, const AudioData& audio) {
    SF_INFO sfinfo;
    sfinfo.samplerate = audio.sampleRate;
    sfinfo.channels = audio.channels;
    sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_FLOAT;

    SNDFILE* file = sf_open(filename.c_str(), SFM_WRITE, &sfinfo);
    if (!file) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    sf_count_t written = sf_write_float(file, audio.samples.data(), audio.samples.size());
    if (written != static_cast<sf_count_t>(audio.samples.size())) {
        sf_close(file);
        throw std::runtime_error("Failed to write all samples to: " + filename);
    }

    sf_close(file);
}