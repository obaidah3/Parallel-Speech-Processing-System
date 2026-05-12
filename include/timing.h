#ifndef TIMING_H
#define TIMING_H

#include <chrono>
#include <string>

// Timing utilities
class Timer {
public:
    Timer();
    void start();
    void stop();
    double elapsedSeconds() const;
    double elapsedMilliseconds() const;

private:
    std::chrono::high_resolution_clock::time_point startTime_;
    std::chrono::high_resolution_clock::time_point endTime_;
};

// MPI timing
double getMpiTime();

// Benchmarking functions
void saveBenchmarkResults(const std::string& filename, double sequentialTime, double parallelTime, int numProcesses, int numThreads);

#endif