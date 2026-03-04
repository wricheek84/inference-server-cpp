#pragma once
#include <vector>
#include <thread>
#include <iostream>
#include "simple_queue.hpp"
#include <chrono>
class ThreadPool {
private:
    std::vector<std::thread> workers;
    SimpleQueue<int> &queue;
   
    void worker_loop(int id) {
        while (true) {
            int task = queue.pop();
            if (task == -1) {
                std::cout << "Worker " << id << " received shutdown signal. Exiting." << std::endl;
                break;
            }
            std::cout << "Worker " << id << " processing task " << task << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
public:
    ThreadPool(int num_threads, SimpleQueue<int>& q) : queue(q){
        
        for (int i = 0; i < num_threads; ++i) {
            workers.emplace_back(&ThreadPool::worker_loop, this, i);
        }

    }
    ~ThreadPool() {
        for(int i=0;i<workers.size();i++){
        queue.push(-1);

        }
        for (auto& worker : workers) {
            worker.join();
        }
    }    

};