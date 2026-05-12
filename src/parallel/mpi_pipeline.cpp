#include <mpi.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include "audio.h"
#include "parallel.h"

// Function to process audio in parallel using MPI and OpenMP
void mpiProcessAudio(const std::string& inputFile, const std::string& outputFile) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double startTime = MPI_Wtime();

    std::vector<float> samples;
    int sampleRate = 0;

    if (rank == 0) {
        // Read audio
        auto [samps, sr] = readWavMono(inputFile);
        samples = samps;
        sampleRate = sr;

        // Preprocessing
        normalizeAudio(samples);
        removeDcOffset(samples);
        applyPreEmphasis(samples);

        std::cout << "Rank 0: Read " << samples.size() << " samples at " << sampleRate << " Hz" << std::endl;
    }

    // Broadcast sample rate
    MPI_Bcast(&sampleRate, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Determine chunk sizes
    int totalSamples = samples.size();
    MPI_Bcast(&totalSamples, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int chunkSize = totalSamples / (size - 1);  // Assuming rank 0 doesn't process
    int remainder = totalSamples % (size - 1);

    std::vector<int> sendCounts(size - 1, chunkSize);
    for (int i = 0; i < remainder; ++i) {
        sendCounts[i]++;
    }

    std::vector<int> displacements(size - 1, 0);
    for (int i = 1; i < size - 1; ++i) {
        displacements[i] = displacements[i - 1] + sendCounts[i - 1];
    }

    if (rank == 0) {
        // Send chunks to workers
        for (int worker = 1; worker < size; ++worker) {
            int count = sendCounts[worker - 1];
            int disp = displacements[worker - 1];
            MPI_Send(&samples[disp], count, MPI_FLOAT, worker, 0, MPI_COMM_WORLD);
        }
    } else {
        // Workers receive chunks
        int myCount = sendCounts[rank - 1];
        std::vector<float> myChunk(myCount);
        MPI_Recv(myChunk.data(), myCount, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // Process chunk with OpenMP
        parallelProcessSamples(myChunk);

        // Send back
        MPI_Send(myChunk.data(), myCount, MPI_FLOAT, 0, 1, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        // Receive processed chunks
        std::vector<float> processedSamples(totalSamples);
        for (int worker = 1; worker < size; ++worker) {
            int count = sendCounts[worker - 1];
            int disp = displacements[worker - 1];
            MPI_Recv(&processedSamples[disp], count, MPI_FLOAT, worker, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Write output
        writeWavMono(outputFile, processedSamples, sampleRate);

        double endTime = MPI_Wtime();
        double parallelTime = endTime - startTime;

        // Estimate sequential time (simple approximation)
        double sequentialTime = parallelTime * (size - 1);  // Rough estimate

        std::cout << "Parallel time: " << parallelTime << " seconds" << std::endl;
        std::cout << "Estimated sequential time: " << sequentialTime << " seconds" << std::endl;

        // Save timing results
        std::ofstream timingFile("data/results/timing.txt");
        timingFile << "Parallel time: " << parallelTime << std::endl;
        timingFile << "Estimated sequential time: " << sequentialTime << std::endl;
        timingFile.close();
    }
}