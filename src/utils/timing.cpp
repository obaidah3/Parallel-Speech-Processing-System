#include "timing.h"
#include <mpi.h>

// Timer implementation
Timer::Timer() : startTime_(), endTime_() {}

void Timer::start() {
    startTime_ = std::chrono::high_resolution_clock::now();
}

void Timer::stop() {
    endTime_ = std::chrono::high_resolution_clock::now();
}

double Timer::elapsedSeconds() const {
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime_ - startTime_);
    return duration.count() / 1e6;
}

double Timer::elapsedMilliseconds() const {
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(endTime_ - startTime_);
    return duration.count() / 1e3;
}

// MPI timing
double getMpiTime() {
    return MPI_Wtime();
}

// Benchmarking
void saveBenchmarkResults(const std::string& filename, double sequentialTime, double parallelTime, int numProcesses, int numThreads) {
    std::ofstream file(filename);
    file << "Sequential Time (s),Parallel Time (s),Speedup,Efficiency,Processes,Threads\n";
    double speedup = sequentialTime > 0 ? sequentialTime / parallelTime : 0.0;
    double efficiency = speedup / (numProcesses * numThreads);
    file << sequentialTime << "," << parallelTime << "," << speedup << "," << efficiency << "," << numProcesses << "," << numThreads << "\n";
    file.close();
}