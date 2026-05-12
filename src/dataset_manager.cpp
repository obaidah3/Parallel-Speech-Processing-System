#include "dataset_manager.h"

#include <algorithm>
#include <cctype>

std::vector<fs::path> DatasetScanner::scan(const fs::path& datasetRoot) const {
    std::vector<fs::path> files;
    if (!fs::exists(datasetRoot) || !fs::is_directory(datasetRoot)) {
        return files;
    }

    for (const auto& entry : fs::recursive_directory_iterator(datasetRoot)) {
        if (!entry.is_regular_file()) {
            continue;
        }

        auto extension = entry.path().extension().string();
        std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });

        if (extension == ".wav") {
            files.push_back(entry.path());
        }
    }

    std::sort(files.begin(), files.end());
    return files;
}

std::vector<AudioTask> DatasetManager::buildTasks(const fs::path& datasetRoot, const fs::path& outputRoot, const std::string& mode) const {
    std::vector<AudioTask> tasks;
    DatasetScanner scanner;
    auto audioFiles = scanner.scan(datasetRoot);
    tasks.reserve(audioFiles.size());

    for (const auto& inputPath : audioFiles) {
        AudioTask task;
        task.inputPath = inputPath;
        task.relativePath = fs::relative(inputPath, datasetRoot);
        task.outputWavPath = outputRoot / "results" / mode / task.relativePath;
        task.outputFeaturePath = outputRoot / "features" / mode / task.relativePath;
        task.outputFeaturePath.replace_extension(".csv");
        tasks.push_back(std::move(task));
    }

    return tasks;
}
