#ifndef PARALLEL_H
#define PARALLEL_H

#include <vector>
#include <string>



// Sequential processing
void runSequentialPipeline(const std::string& inputFile, const std::string& outputFile, const std::string& featuresFile);

// OpenMP parallel processing functions
void parallelProcessSamples(std::vector<float>& samples);
void ompStft(std::vector<std::vector<std::complex<float>>>& spectrogram, const std::vector<float>& samples, int fftSize, int hopSize);
void ompFiltering(std::vector<std::vector<std::complex<float>>>& spectrogram, int sampleRate);
void ompFeatureExtraction(const std::vector<std::vector<std::complex<float>>>& spectrogram, int sampleRate, const std::string& outputFile);

// MPI pipeline functions
void mpiProcessAudio(const std::string& inputFile, const std::string& outputFile);
void mpiStftPipeline(const std::vector<float>& samples, int sampleRate, std::vector<std::vector<std::complex<float>>>& spectrogram, int fftSize, int hopSize);
void mpiFilteringPipeline(std::vector<std::vector<std::complex<float>>>& spectrogram, int sampleRate);

// Chunk management
std::vector<std::vector<float>> splitIntoChunks(const std::vector<float>& data, int numChunks);
std::vector<float> mergeChunks(const std::vector<std::vector<float>>& chunks);

#endif