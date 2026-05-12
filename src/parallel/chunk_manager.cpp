#include "parallel.h"
#include <vector>
#include <algorithm>

// Split data into chunks for distribution
std::vector<std::vector<float>> splitIntoChunks(const std::vector<float>& data, int numChunks) {
    std::vector<std::vector<float>> chunks;
    int chunkSize = data.size() / numChunks;
    int remainder = data.size() % numChunks;

    int start = 0;
    for (int i = 0; i < numChunks; ++i) {
        int size = chunkSize + (i < remainder ? 1 : 0);
        chunks.push_back(std::vector<float>(data.begin() + start, data.begin() + start + size));
        start += size;
    }

    return chunks;
}

// Merge chunks back into single vector
std::vector<float> mergeChunks(const std::vector<std::vector<float>>& chunks) {
    std::vector<float> merged;
    for (const auto& chunk : chunks) {
        merged.insert(merged.end(), chunk.begin(), chunk.end());
    }
    return merged;
}