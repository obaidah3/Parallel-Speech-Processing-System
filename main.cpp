#include "compiler_workaround.h"

#include <mpi.h>
#include <iostream>
#include <string>
#include "parallel.h"
#include "timing.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <mode> [num_processes] [num_threads]" << std::endl;
        std::cout << "Modes: sequential, omp, mpi, hybrid" << std::endl;
        std::cout << "Examples:" << std::endl;
        std::cout << "  " << argv[0] << " sequential" << std::endl;
        std::cout << "  " << argv[0] << " omp 4" << std::endl;
        std::cout << "  mpirun -np 2 " << argv[0] << " mpi" << std::endl;
        std::cout << "  mpirun -np 2 " << argv[0] << " hybrid 2 4" << std::endl;
        return 1;
    }

    std::string mode = argv[1];

    if (mode == "sequential") {
        // Sequential mode - no MPI, no OpenMP
        std::cout << "Running Sequential mode..." << std::endl;
        runSequentialPipeline("data/input.wav", "data/output_sequential.wav", "data/results/features_sequential.csv");
    } else if (mode == "omp") {
        // OpenMP only mode - no MPI
        int numThreads = 4; // default
        if (argc > 2) numThreads = std::stoi(argv[2]);
        std::cout << "Running OpenMP mode with " << numThreads << " threads..." << std::endl;

        // Set OpenMP threads
        #pragma omp parallel num_threads(numThreads)
        {
            // This sets the number of threads
        }

        runOmpPipeline("data/input.wav", "data/output_omp.wav", "data/results/features_omp.csv");
    } else if (mode == "mpi") {
        // MPI only mode - initialize MPI, process with MPI but sequential per rank
        MPI_Init(&argc, &argv);
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        if (rank == 0) {
            std::cout << "Running MPI-only mode with " << size << " processes" << std::endl;
        }

        mpiProcessAudio("data/input.wav", "data/output_mpi.wav");

        MPI_Finalize();
    } else if (mode == "hybrid") {
        // Hybrid MPI + OpenMP mode
        MPI_Init(&argc, &argv);
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        int numThreads = 2; // default
        if (argc > 3) numThreads = std::stoi(argv[3]);

        if (rank == 0) {
            std::cout << "Running Hybrid mode with " << size << " processes and " << numThreads << " threads per process" << std::endl;
        }

        // Set OpenMP threads
        #pragma omp parallel num_threads(numThreads)
        {
            // Set number of threads
        }

        mpiProcessAudio("data/input.wav", "data/output_hybrid.wav");

        MPI_Finalize();
    } else {
        std::cerr << "Unknown mode: " << mode << std::endl;
        std::cout << "Available modes: sequential, omp, mpi, hybrid" << std::endl;
        return 1;
    }

    return 0;
}