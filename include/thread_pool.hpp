#pragma once
#include <vector>
#include <thread>
#include <iostream>
#include "simple_queue.hpp"
#include <chrono>
#include <onnxruntime_cxx_api.h>
#include "tensor.hpp"
#include<atomic>



class ThreadPool {
private:
    std::vector<std::thread> workers;
    Ort::Env env{ORT_LOGGING_LEVEL_FATAL, "InferenceServer"};
    std::unique_ptr<Ort::Session> session;
    SimpleQueue<int> &queue;
    std::atomic<int> active_workers{0};
    std::atomic<int> total_processed{0};
    
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
            Tensor my_tensor(batch);
            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value input_tensor = Ort::Value::CreateTensor<int64_t>(memory_info, my_tensor.data.data(), my_tensor.data.size(), my_tensor.shape.data(), my_tensor.shape.size());
            const char* input_names[] = {"input"};
            const char* output_names[] = {"output"};
            std::cout << "Worker " << id << " feeding batch of " << batch.size() << " to AI model..." << std::endl;
            active_workers++;
            auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
            active_workers--;
            total_processed += batch.size();
            float* floatarr = output_tensors.front().GetTensorMutableData<float>();
            std::cout << "Worker " << id << " got AI output: " << floatarr[0] << std::endl;




            
            
            
        }
    }
public:
    ThreadPool(int num_threads, SimpleQueue<int>& q) : queue(q){
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        try {
            
            session = std::make_unique<Ort::Session>(env, L"C:\\Users\\wrich\\inference-server-cpp\\build\\Debug\\model.onnx", session_options);
        } catch (const Ort::Exception& e) {
            std::cerr << "\n[CRITICAL AI ERROR]: " << e.what() << "\n" << std::endl;
            exit(1);
        }
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
    int get_active_workers() const { return active_workers.load(); }
    int get_total_processed() const { return total_processed.load(); }   
};