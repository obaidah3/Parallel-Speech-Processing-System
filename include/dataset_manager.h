#ifndef DATASET_MANAGER_H
#define DATASET_MANAGER_H

#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct AudioTask {
    fs::path inputPath;
    fs::path relativePath;
    fs::path outputWavPath;
    fs::path outputFeaturePath;
};

class DatasetScanner {
public:
    std::vector<fs::path> scan(const fs::path& datasetRoot) const;
};

class DatasetManager {
public:
    std::vector<AudioTask> buildTasks(const fs::path& datasetRoot, const fs::path& outputRoot, const std::string& mode) const;
};

#endif // DATASET_MANAGER_H
