#include <mpi.h>
#include <iostream>
#include <string>
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: test mode" << std::endl;
        return 1;
    }
    std::string mode = argv[1];
    std::cout << "Mode: " << mode << std::endl;
    return 0;
}
