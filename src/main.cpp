#include <iostream>
#include <thread>

int main() {
    unsigned int n = std::thread::hardware_concurrency();
    std::cout << "Hello! Your CPU has " << n << " threads available.\n";
    std::cout << "System is ready for Phase 1.\n";
    return 0;
}