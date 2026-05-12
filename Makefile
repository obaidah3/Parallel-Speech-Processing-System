CXX = g++
CXXFLAGS = -std=c++17 -Wall -O3 -fopenmp -Iinclude -I/usr/include/x86_64-linux-gnu/mpi
LDFLAGS = -lsndfile -lfftw3 -lfftw3f -fopenmp -lmpi
SOURCES = main.cpp dataset_manager.cpp wav_reader.cpp wav_writer.cpp preprocessing.cpp fft_utils.cpp stft.cpp istft.cpp filters.cpp feature_extraction.cpp omp_pipeline.cpp mpi_pipeline.cpp chunk_manager.cpp sequential_pipeline.cpp timing.cpp logger.cpp helpers.cpp
BUILD_DIR = build
OBJECTS = $(addprefix $(BUILD_DIR)/, $(SOURCES:.cpp=.o))
EXECUTABLE = audio_app

# VPATH for source files
.DEFAULT_GOAL := all
VPATH = .:src:src/io:src/dsp:src/parallel:src/sequential:src/utils

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Compile objects to build/
$(BUILD_DIR)/%.o: %.cpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

all: $(BUILD_DIR) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $@ $(LDFLAGS)

clean:
	rm -rf $(BUILD_DIR) $(EXECUTABLE)

.PHONY: all clean
