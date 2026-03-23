#pragma once
#include <vector>
#include <thread>
#include <iostream>
#include "simple_queue.hpp"
#include <chrono>
#include <onnxruntime_cxx_api.h>
#include "tensor.hpp"
#include<atomic>
struct InferenceRequest {
    
    int64_t token; 
    int64_t mask; 
    std::chrono::steady_clock::time_point arrival_time;
};



class ThreadPool {
private:
    std::vector<std::thread> workers;
    Ort::Env env{ORT_LOGGING_LEVEL_FATAL, "InferenceServer"};
    std::unique_ptr<Ort::Session> session;
    SimpleQueue<InferenceRequest> &queue;
    std::atomic<int> active_workers{0};
    std::atomic<int> total_processed{0};
    std::atomic<int> total_batches{0};
    std::atomic<long long> total_latency_ms{0};
    
    void worker_loop(int id) {
        while (true) {
            std::vector<InferenceRequest> requests = queue.pop_batch(32,5);
            
            if (requests.empty()) { 
                continue; 
            }
            
            if(requests[0].token==-1){
                std::cout<<"Worker "<<id<<" shutting down!"<<std::endl;
                break;
            }
            std::vector<int64_t> batch_tokens; 
            std::vector<int64_t> batch_masks; 

            for (const auto& req : requests) {

                batch_tokens.push_back(req.token);
                batch_masks.push_back(req.mask); 
            }
            std::cout<<"Worker "<<id<<" processing a batch of "<<requests.size()<<" tokens!"<<std::endl;
            int64_t batch_size = batch_tokens.size();
            std::vector<int64_t> shape = { batch_size, 1 };

            Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

            
            Ort::Value ids_tensor = Ort::Value::CreateTensor<int64_t>(
                memory_info, batch_tokens.data(), batch_tokens.size(), shape.data(), shape.size());
            
            Ort::Value mask_tensor = Ort::Value::CreateTensor<int64_t>(
                memory_info, batch_masks.data(), batch_masks.size(), shape.data(), shape.size());

            
            const char* input_names[] = {"input_ids", "attention_mask"};
            const char* output_names[] = {"logits"};

            
            Ort::Value input_tensors[] = {std::move(ids_tensor), std::move(mask_tensor)};
            
            active_workers++;
            
            auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_names, input_tensors, 2, output_names, 1);
            
            active_workers--;
            total_processed += requests.size();
            total_batches++;
            auto now = std::chrono::steady_clock::now();
            long long batch_latency = 0;
            for (const auto& req : requests) {
                batch_latency += std::chrono::duration_cast<std::chrono::milliseconds>(now - req.arrival_time).count();
            }
            total_latency_ms += batch_latency;
            float* floatarr = output_tensors.front().GetTensorMutableData<float>();
            std::cout << "Worker " << id << " got AI output: " << floatarr[0] << std::endl;




            
            
            
        }
    }
public:
    ThreadPool(int num_threads, SimpleQueue<InferenceRequest>& q) : queue(q){
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        try {
            
            session = std::make_unique<Ort::Session>(env, L"C:\\Users\\wrich\\inference-server-cpp\\onnx\\model_quantized.onnx", session_options);
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
            queue.push(InferenceRequest{-1,0,std::chrono::steady_clock::now()});
        }
        for (auto& worker : workers) {
            worker.join();
        }
    } 
    int get_active_workers() const { return active_workers.load(); }
    int get_total_processed() const { return total_processed.load(); }   
    int get_total_batches() const { return total_batches.load(); }
    long long get_total_latency_ms() const { return total_latency_ms.load(); }
};