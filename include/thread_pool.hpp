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
            std::vector<int> batch=queue.pop_batch(32,5);
            
            if(batch.empty()){
                continue;
            }
            if(batch[0]==-1){
                std::cout<<"Worker "<<id<<" shutting down!"<<std::endl;
                break;
            }
            std::cout<<"Worker "<<id<<" processing a batch of "<<batch.size()<<" tokens!"<<std::endl;
            for(int i=0;i<batch.size();i++){
                std::cout<<"Worker "<<id<<" processing task "<<batch[i]<<std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }



            
            
            
        }
    }
public:
    ThreadPool(int num_threads, SimpleQueue<int>& q) : queue(q){
        for (int i = 0; i < num_threads; ++i) {
            workers.emplace_back(&ThreadPool::worker_loop, this, i);
        }
    }
    
    ~ThreadPool() {
        // This is perfectly fine as is! It will push multiple -1s into the queue.
        for(int i=0;i<workers.size();i++){
            queue.push(-1);
        }
        for (auto& worker : workers) {
            worker.join();
        }
    }    
};