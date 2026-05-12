#include <iostream>
#include <string>

// Simple logger
void logMessage(const std::string& message) {
    std::cout << "[LOG] " << message << std::endl;
}

void logError(const std::string& message) {
    std::cerr << "[ERROR] " << message << std::endl;
}