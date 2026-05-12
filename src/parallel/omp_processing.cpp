#include <vector>
#include <iostream>
#include <omp.h>

// Parallel processing of audio samples using OpenMP
// Demonstrates parallel loop execution with thread identification
// For testing purposes, applies an identity operation (multiply by 1.0)
// Prints thread IDs for the first few samples to show parallelism
void parallelProcessSamples(std::vector<float>& samples) {
    // Parallel for loop over all samples
    #pragma omp parallel for
    for (size_t i = 0; i < samples.size(); ++i) {
        // Identity operation for demonstration
        // In a real application, this could be filtering, FFT, etc.
        samples[i] = samples[i] * 1.0f;

        // Print thread information for first 10 samples to avoid excessive output
        // Uses critical section to prevent interleaved output
        if (i < 10) {
            #pragma omp critical
            {
                std::cout << "Thread " << omp_get_thread_num() 
                          << " processing sample " << i << std::endl;
            }
        }
    }
}