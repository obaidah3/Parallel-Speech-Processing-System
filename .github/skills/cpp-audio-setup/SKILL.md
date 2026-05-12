---
name: cpp-audio-setup
description: Set up a modern C++17 audio signal processing project with WAV reading/writing using libsndfile. Creates necessary files, implements audio I/O functions, updates main.cpp, and configures Makefile.
---

# C++ Audio Project Setup

This skill sets up a complete C++17 audio project for processing WAV files using libsndfile.

## Prerequisites

- libsndfile installed (e.g., `sudo apt install libsndfile1-dev` on Ubuntu)
- g++ with C++17 support

## Steps

1. Create `include/audio.h` with AudioData struct and function declarations.

2. Create `src/audio.cpp` with implementations of readWav and writeWav.

3. Update `main.cpp` to read from `data/input.wav`, print metadata, write to `data/output.wav`, with try/catch.

4. Update `Makefile` with proper compilation flags, includes, sources, and linking.

## Files Created/Modified

### include/audio.h

```cpp
#ifndef AUDIO_H
#define AUDIO_H

#include <vector>
#include <string>

struct AudioData {
    std::vector<float> samples;
    int sampleRate;
    int channels;
};

AudioData readWav(const std::string& filename);
void writeWav(const std::string& filename, const AudioData& audio);

#endif
```

### src/audio.cpp

```cpp
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
    if (read != audio.samples.size()) {
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
    if (written != audio.samples.size()) {
        sf_close(file);
        throw std::runtime_error("Failed to write all samples to: " + filename);
    }

    sf_close(file);
}
```

### main.cpp

```cpp
#include "audio.h"
#include <iostream>

int main() {
    try {
        AudioData audio = readWav("data/input.wav");
        std::cout << "Sample rate: " << audio.sampleRate << std::endl;
        std::cout << "Channels: " << audio.channels << std::endl;
        std::cout << "Total samples: " << audio.samples.size() << std::endl;

        writeWav("data/output.wav", audio);
        std::cout << "Audio written to data/output.wav" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
```

### Makefile

```makefile
CXX = g++
CXXFLAGS = -std=c++17 -Wall -O2 -Iinclude
LDFLAGS = -lsndfile
SOURCES = main.cpp src/audio.cpp
OBJECTS = $(SOURCES:.cpp=.o)
EXECUTABLE = audio_app

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $@ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJECTS) $(EXECUTABLE)

.PHONY: all clean
```

This skill produces a complete, production-ready C++ audio project setup.

Example prompt: "Set up a C++ audio project with libsndfile for WAV processing."

Related customizations: Perhaps a skill for audio effects or FFT processing.