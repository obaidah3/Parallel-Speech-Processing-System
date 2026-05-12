#include "compiler_workaround.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <mpi.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "dataset_manager.h"
#include "parallel.h"
#include "timing.h"

namespace fs = std::filesystem;

static void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " --dataset <path> --mode <sequential|omp|mpi|hybrid> --output <path> [--threads <n>]" << std::endl;
    std::cout << "Example: " << programName << " --dataset dataset --mode omp --output output --threads 4" << std::endl;
}

int main(int argc, char* argv[]) {
    std::string datasetPath;
    std::string mode;
    std::string outputPath;
    int numThreads = 4;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--dataset" && i + 1 < argc) {
            datasetPath = argv[++i];
        } else if (arg == "--mode" && i + 1 < argc) {
            mode = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            outputPath = argv[++i];
        } else if (arg == "--threads" && i + 1 < argc) {
            numThreads = std::stoi(argv[++i]);
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }

    if (datasetPath.empty() || mode.empty() || outputPath.empty()) {
        printUsage(argv[0]);
        return 1;
    }

    fs::path datasetRoot(datasetPath);
    fs::path outputRoot(outputPath);

    if (!fs::exists(datasetRoot) || !fs::is_directory(datasetRoot)) {
        std::cerr << "Dataset directory does not exist: " << datasetRoot << std::endl;
        return 1;
    }

    if (mode == "mpi" || mode == "hybrid") {
        MPI_Init(&argc, &argv);

        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) {
            std::cout << "Running " << mode << " MPI dataset mode" << std::endl;
        }

        mpiProcessDataset(datasetRoot.string(), outputRoot.string(), mode, numThreads);

        MPI_Finalize();
        return 0;
    }

    DatasetManager manager;
    auto tasks = manager.buildTasks(datasetRoot, outputRoot, mode);

    if (tasks.empty()) {
        std::cerr << "No WAV files found in dataset: " << datasetRoot << std::endl;
        return 1;
    }

    fs::create_directories(outputRoot);
    std::ofstream benchmarkFile(outputRoot / "benchmark_results.csv");
    if (!benchmarkFile) {
        std::cerr << "Failed to create benchmark file: " << (outputRoot / "benchmark_results.csv") << std::endl;
        return 1;
    }
    benchmarkFile << "mode,file,relative_path,duration_seconds\n";

    for (const auto& task : tasks) {
        Timer timer;
        timer.start();

        if (mode == "sequential") {
            runSequentialPipeline(task.inputPath.string(), task.outputWavPath.string(), task.outputFeaturePath.string());
        } else if (mode == "omp") {
            #ifdef _OPENMP
            omp_set_num_threads(numThreads);
            #endif
            runOmpPipeline(task.inputPath.string(), task.outputWavPath.string(), task.outputFeaturePath.string());
        } else {
            std::cerr << "Unknown mode: " << mode << std::endl;
            return 1;
        }

        timer.stop();
        double duration = timer.elapsedSeconds();
        benchmarkFile << mode << "," << task.inputPath.string() << "," << task.relativePath.string() << "," << duration << "\n";
    }

    benchmarkFile.close();
    std::cout << "Dataset processing complete. Benchmark results written to " << (outputRoot / "benchmark_results.csv") << std::endl;
    return 0;
}
