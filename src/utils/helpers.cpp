#include <vector>
#include <algorithm>
#include <cmath>

// Utility functions
float clamp(float value, float min, float max) {
    return std::max(min, std::min(max, value));
}

std::vector<float> linspace(float start, float end, int num) {
    std::vector<float> result(num);
    float step = (end - start) / (num - 1);
    for (int i = 0; i < num; ++i) {
        result[i] = start + i * step;
    }
    return result;
}

float dbToLinear(float db) {
    return std::pow(10.0f, db / 20.0f);
}

float linearToDb(float linear) {
    return 20.0f * std::log10(linear);
}