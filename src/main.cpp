#include <iostream>
#include <thread>
#include "simple_queue.hpp"
#include "thread_pool.hpp"

SimpleQueue<int> order_queue;

int main() {
    unsigned int n = std::thread::hardware_concurrency();
    std::cout << "Starting " << n << " worker threads." << std::endl;

    
    ThreadPool pool(n, order_queue);

    
    for (int i = 0; i < 20; i++) {
        order_queue.push(i);
    }

    std::cout << "All orders submitted." << std::endl;

    return 0;
}