#include "dataset_manager.h"
#include "parallel.h"
#include "audio.h"
#include "audio_features.h"
#include "timing.h"
#include <mpi.h>
#include <filesystem>
#include <sstream>
#include <fstream>
#include <iostream>
#include <tuple>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace fs = std::filesystem;
static bool validateProcessing(const AudioTask& task, double tolerance = 1e-6) {
    // Read original and processed audio
    auto [original, origRate] = readWavMono(task.inputPath.string());
    auto [processed, procRate] = readWavMono(task.outputWavPath.string());

    if (origRate != procRate) {
        std::cerr << "Sample rate mismatch for " << task.inputPath << std::endl;
        return false;
    }

    // Basic validation: check that output is not silence and has reasonable length
    if (processed.empty()) {
        std::cerr << "Empty output for " << task.inputPath << std::endl;
        return false;
    }

    // Check RMS levels are reasonable
    float origRMS = calculateRMS(original);
    float procRMS = calculateRMS(processed);

    if (procRMS < tolerance) {
        std::cerr << "Output appears to be silence for " << task.inputPath << std::endl;
        return false;
    }

    // Check that features file exists and has content
    std::ifstream featuresFile(task.outputFeaturePath);
    if (!featuresFile.good()) {
        std::cerr << "Features file missing for " << task.inputPath << std::endl;
        return false;
    }

    std::string line;
    bool hasData = false;
    while (std::getline(featuresFile, line)) {
        if (!line.empty() && line.find("Frame") == std::string::npos) {
            hasData = true;
            break;
        }
    }

    if (!hasData) {
        std::cerr << "Features file empty for " << task.inputPath << std::endl;
        return false;
    }

    return true;
}

namespace fs = std::filesystem;

static void sendString(int dest, int tag, const std::string& payload) {
    int length = static_cast<int>(payload.size());
    MPI_Send(&length, 1, MPI_INT, dest, tag, MPI_COMM_WORLD);
    if (length > 0) {
        MPI_Send(payload.data(), length, MPI_CHAR, dest, tag + 100, MPI_COMM_WORLD);
    }
}

static std::string receiveString(int source, int tag) {
    int length = 0;
    MPI_Recv(&length, 1, MPI_INT, source, tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    std::string payload;
    if (length > 0) {
        payload.resize(length);
        MPI_Recv(payload.data(), length, MPI_CHAR, source, tag + 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    return payload;
}

static std::string serializeTaskList(const std::vector<fs::path>& paths, const fs::path& datasetRoot, int rank) {
    std::ostringstream oss;
    for (const auto& path : paths) {
        auto relativePath = fs::relative(path, datasetRoot);
        if (relativePath.empty()) {
            relativePath = path.filename();
        }
        oss << path.string() << '\n';
        oss << relativePath.string() << '\n';
        oss << rank << '\n';
    }
    return oss.str();
}

static std::vector<AudioTask> deserializeRankTasks(const std::string& payload, const fs::path& datasetRoot, const fs::path& outputRoot, const std::string& mode) {
    std::vector<AudioTask> tasks;
    std::istringstream iss(payload);
    std::string inputPath;
    std::string relativePath;
    std::string rankString;

    while (std::getline(iss, inputPath) && std::getline(iss, relativePath) && std::getline(iss, rankString)) {
        int rank = std::stoi(rankString);
        AudioTask task;
        task.inputPath = inputPath;
        task.relativePath = relativePath;
        task.outputWavPath = outputRoot / "results" / mode / ("rank_" + std::to_string(rank)) / fs::path(relativePath);
        task.outputFeaturePath = outputRoot / "features" / mode / ("rank_" + std::to_string(rank)) / fs::path(relativePath);
        task.outputFeaturePath.replace_extension(".csv");
        tasks.push_back(std::move(task));
    }
    return tasks;
}

static std::string formatBenchmarkLine(int rank, const std::string& mode, const AudioTask& task, double duration) {
    std::error_code ec;
    uintmax_t fileSize = fs::file_size(task.inputPath, ec);
    if (ec) fileSize = 0;

    std::ostringstream oss;
    oss << rank << "," << mode << "," << task.inputPath.string() << "," << task.relativePath.string() << "," << duration << "," << fileSize;
    return oss.str();
}

static void writeBenchmarkFile(const std::string& outputRoot, const std::string& mode, const std::vector<std::string>& rows, int rank = 0) {
    fs::path benchmarkPath = fs::path(outputRoot) / ("benchmark_results_" + mode + "_rank" + std::to_string(rank) + ".csv");
    fs::create_directories(benchmarkPath.parent_path());
    std::ofstream benchmarkFile(benchmarkPath);
    benchmarkFile << "rank,mode,file,relative_path,duration_seconds,file_size_bytes\n";
    for (const auto& row : rows) {
        benchmarkFile << row << "\n";
    }
}

static void writeSummaryFile(const std::string& outputRoot, const std::string& mode, const std::vector<std::tuple<int, std::string, int, double>>& summaries) {
    fs::path summaryPath = fs::path(outputRoot) / ("benchmark_summary_" + mode + ".csv");
    fs::create_directories(summaryPath.parent_path());
    std::ofstream summaryFile(summaryPath);
    summaryFile << "rank,mode,files_processed,total_seconds,throughput_files_per_sec\n";
    for (const auto& summary : summaries) {
        int rank;
        std::string modeString;
        int count;
        double totalSeconds;
        std::tie(rank, modeString, count, totalSeconds) = summary;
        double throughput = count > 0 ? static_cast<double>(count) / totalSeconds : 0.0;
        summaryFile << rank << "," << modeString << "," << count << "," << totalSeconds << "," << throughput << "\n";
    }
}

static void ensureTaskDirectories(const AudioTask& task) {
    fs::create_directories(task.outputWavPath.parent_path());
    fs::create_directories(task.outputFeaturePath.parent_path());
}

static std::pair<std::vector<std::string>, std::tuple<int, std::string, int, double>> processLocalTasks(const std::vector<AudioTask>& tasks, const std::string& mode, int numThreads, int rank) {
    std::vector<std::string> benchmarkRows;
    double totalSeconds = 0.0;

    for (const auto& task : tasks) {
        ensureTaskDirectories(task);
        Timer timer;
        timer.start();

        if (mode == "mpi") {
            runSequentialPipeline(task.inputPath.string(), task.outputWavPath.string(), task.outputFeaturePath.string());
        } else if (mode == "hybrid") {
#ifdef _OPENMP
            omp_set_num_threads(numThreads);
#endif
            runOmpPipeline(task.inputPath.string(), task.outputWavPath.string(), task.outputFeaturePath.string());
        }

        timer.stop();
        double duration = timer.elapsedSeconds();
        totalSeconds += duration;
        benchmarkRows.push_back(formatBenchmarkLine(rank, mode, task, duration));

        // Validate processing results
        if (!validateProcessing(task)) {
            std::cerr << "Validation failed for task: " << task.inputPath << std::endl;
        }
    }

    return {benchmarkRows, {rank, mode, static_cast<int>(tasks.size()), totalSeconds}};
}

void mpiProcessDataset(const std::string& datasetRootStr, const std::string& outputRootStr, const std::string& mode, int numThreads) {
    fs::path datasetRoot(datasetRootStr);
    fs::path outputRoot(outputRootStr);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    DatasetScanner scanner;
    if (rank == 0) {
        auto allFiles = scanner.scan(datasetRoot);
        if (allFiles.empty()) {
            for (int dest = 1; dest < size; ++dest) {
                sendString(dest, 0, std::string());
            }
            writeBenchmarkFile(outputRootStr, mode, {}, 0);
            writeSummaryFile(outputRootStr, mode, {});
            return;
        }

        // Size-aware load balancing: distribute files by total size rather than count
        std::vector<std::vector<fs::path>> filesByRank(size);
        std::vector<uintmax_t> rankSizes(size, 0);

        // Sort files by size (largest first) for better load balancing
        std::vector<std::pair<uintmax_t, fs::path>> filesWithSizes;
        for (const auto& file : allFiles) {
            std::error_code ec;
            uintmax_t size = fs::file_size(file, ec);
            if (!ec) {
                filesWithSizes.emplace_back(size, file);
            } else {
                // If we can't get size, assign a default
                filesWithSizes.emplace_back(1024 * 1024, file); // 1MB default
            }
        }

        // Sort by size descending
        std::sort(filesWithSizes.rbegin(), filesWithSizes.rend());

        // Distribute to ranks using greedy assignment to minimize max load
        for (const auto& [size, file] : filesWithSizes) {
            // Find rank with current smallest total size
            size_t minRank = 0;
            uintmax_t minSize = rankSizes[0];
            for (size_t r = 1; r < size; ++r) {
                if (rankSizes[r] < minSize) {
                    minSize = rankSizes[r];
                    minRank = r;
                }
            }
            filesByRank[minRank].push_back(file);
            rankSizes[minRank] += size;
        }

        for (int dest = 1; dest < size; ++dest) {
            sendString(dest, 0, serializeTaskList(filesByRank[dest], datasetRoot, dest));
        }

        // Use non-blocking receives for better performance
        std::vector<MPI_Request> rowRequests(size - 1);
        std::vector<MPI_Request> summaryRequests(size - 1);
        std::vector<int> rowLengths(size - 1);
        std::vector<int> summaryLengths(size - 1);
        std::vector<std::string> rowPayloads(size - 1);
        std::vector<std::string> summaryPayloads(size - 1);

        for (int src = 1; src < size; ++src) {
            int idx = src - 1;
            // Non-blocking receive for row length
            MPI_Irecv(&rowLengths[idx], 1, MPI_INT, src, 1, MPI_COMM_WORLD, &rowRequests[idx]);
            // Non-blocking receive for summary length
            MPI_Irecv(&summaryLengths[idx], 1, MPI_INT, src, 2, MPI_COMM_WORLD, &summaryRequests[idx]);
        }

        // Use MPI_Allreduce for global summary statistics to avoid rank 0 bottleneck
        auto [localRows, localSummary] = processLocalTasks(rank0Tasks, mode, numThreads, 0);
        std::vector<std::string> allRows = std::move(localRows);
        std::vector<std::tuple<int, std::string, int, double>> summaries;
        summaries.push_back(localSummary);

        // Gather summaries from all ranks using MPI_Allreduce
        int localCount;
        double localTotalSeconds;
        std::tie(std::ignore, std::ignore, localCount, localTotalSeconds) = localSummary;

        int globalCount = 0;
        double globalTotalSeconds = 0.0;
        MPI_Allreduce(&localCount, &globalCount, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&localTotalSeconds, &globalTotalSeconds, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // Each rank writes its own benchmark file to avoid I/O bottleneck
        writeBenchmarkFile(outputRootStr, mode, allRows, 0);

        // Only rank 0 writes the global summary
        if (rank == 0) {
            std::vector<std::tuple<int, std::string, int, double>> globalSummaries = {
                {0, mode, globalCount, globalTotalSeconds}
            };
            writeSummaryFile(outputRootStr, mode, globalSummaries);
        }
    } else {
        std::string payload = receiveString(0, 0);
        auto tasks = deserializeRankTasks(payload, datasetRoot, outputRoot, mode);
        auto [localRows, localSummary] = processLocalTasks(tasks, mode, numThreads, rank);

        // Each rank writes its own benchmark file
        writeBenchmarkFile(outputRootStr, mode, localRows, rank);

        // Use MPI_Allreduce for global summary
        int localCount;
        double localTotalSeconds;
        std::tie(std::ignore, std::ignore, localCount, localTotalSeconds) = localSummary;

        int globalCount = 0;
        double globalTotalSeconds = 0.0;
        MPI_Allreduce(&localCount, &globalCount, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&localTotalSeconds, &globalTotalSeconds, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        // Only rank 0 writes the global summary
        if (rank == 0) {
            std::vector<std::tuple<int, std::string, int, double>> globalSummaries = {
                {0, mode, globalCount, globalTotalSeconds}
            };
            writeSummaryFile(outputRootStr, mode, globalSummaries);
        }
    }
}
