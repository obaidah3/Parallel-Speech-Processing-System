#ifndef PARALLEL_H
#define PARALLEL_H

#include <vector>
#include <string>
#include <complex>
#include "dsp.h"

// Sequential processing
void runSequentialPipeline(const std::string& inputFile, const std::string& outputFile, const std::string& featuresFile);

// OpenMP parallel processing functions
void parallelProcessSamples(std::vector<float>& samples);
void ompStft(Spectrogram& spectrogram, const std::vector<float>& samples, int fftSize, int hopSize);
void ompFiltering(Spectrogram& spectrogram, int sampleRate);
void runOmpPipeline(const std::string& inputFile, const std::string& outputFile, const std::string& featuresFile);

// MPI dataset processing
void mpiProcessDataset(const std::string& datasetRoot, const std::string& outputRoot, const std::string& mode, int numThreads);

// Chunk management
std::vector<std::vector<float>> splitIntoChunks(const std::vector<float>& data, int numChunks);
std::vector<float> mergeChunks(const std::vector<std::vector<float>>& chunks);

#endif
