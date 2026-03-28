#pragma once
#include <vector>
#include <thread>
#include <iostream>
#include "simple_queue.hpp"
#include <chrono>
#include <onnxruntime_cxx_api.h>
#include "tensor.hpp"
#include <atomic>
#include <mutex>
#include <algorithm>

struct InferenceRequest {
    int64_t token; 
    int64_t mask; 
    std::chrono::steady_clock::time_point arrival_time;
};

struct TelemetrySnapshot {
    int queue_peak;
    int tasks_processed;
    int worker_peak;
    long long worker_active_time_ns;
};

class ThreadPool {
private:
    struct TelemetryWindow {
        std::atomic<int> max_queue_depth{0};
        std::atomic<int> tasks_processed{0};
        std::atomic<int> active_worker_count_peak{0};
        std::atomic<long long> worker_active_time_ns{0};
    };

    std::vector<std::thread> workers;
    std::vector<double> latency_samples; 
    std::mutex samples_mutex;
    Ort::Env env{ORT_LOGGING_LEVEL_FATAL, "InferenceServer"};
    std::unique_ptr<Ort::Session> session;
    SimpleQueue<InferenceRequest>& queue;
    std::atomic<int> active_workers{0};
    std::atomic<int> total_processed{0};
    std::atomic<int> total_batches{0};
    std::atomic<long long> total_latency_ms{0};
    
    TelemetryWindow telemetry;

    void worker_loop(int id) {
        while (true) {
            std::vector<InferenceRequest> requests = queue.pop_batch(32, 25);
            
            int current_q = queue.get_queue_depth();
            int cur_max_q = telemetry.max_queue_depth.load(std::memory_order_relaxed);
            while (current_q > cur_max_q && !telemetry.max_queue_depth.compare_exchange_weak(cur_max_q, current_q, std::memory_order_relaxed)) {}

            if (requests.empty()) { 
                continue; 
            }
            
            if(requests[0].token == -1){
                std::cout << "Worker " << id << " shutting down!" << std::endl;
                break;
            }

            std::vector<int64_t> batch_tokens; 
            std::vector<int64_t> batch_masks; 

            for (const auto& req : requests) {
                batch_tokens.push_back(req.token);
                batch_masks.push_back(req.mask); 
            }

            std::cout << "Worker " << id << " processing a batch of " << requests.size() << " tokens!" << std::endl;
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
            
            int current_active = ++active_workers;
            int cur_peak = telemetry.active_worker_count_peak.load(std::memory_order_relaxed);
            while (current_active > cur_peak && !telemetry.active_worker_count_peak.compare_exchange_weak(cur_peak, current_active, std::memory_order_relaxed)) {}

            auto start_infer = std::chrono::steady_clock::now();
            auto output_tensors = session->Run(Ort::RunOptions{nullptr}, input_names, input_tensors, 2, output_names, 1);
            auto end_infer = std::chrono::steady_clock::now();
            
            active_workers--;

            telemetry.worker_active_time_ns.fetch_add(std::chrono::duration_cast<std::chrono::nanoseconds>(end_infer - start_infer).count(), std::memory_order_relaxed);
            telemetry.tasks_processed.fetch_add((int)requests.size(), std::memory_order_relaxed);

            total_processed += (int)requests.size();
            total_batches++;

            auto now = std::chrono::steady_clock::now();
            long long batch_latency_sum = 0;

            {
                std::lock_guard<std::mutex> lock(samples_mutex);
                for (const auto& req : requests) {
                    double lat = (double)std::chrono::duration_cast<std::chrono::milliseconds>(now - req.arrival_time).count();
                    batch_latency_sum += (long long)lat;
                    
                    latency_samples.push_back(lat);
                    if (latency_samples.size() > 2000) {
                        latency_samples.erase(latency_samples.begin());
                    }
                }
            }
            total_latency_ms += batch_latency_sum;

            float* floatarr = output_tensors.front().GetTensorMutableData<float>();
            std::cout << "Worker " << id << " got AI output: " << floatarr[0] << std::endl;
        }
    }

public:
    struct Percentiles { double p50; double p99; };

    ThreadPool(int num_threads, SimpleQueue<InferenceRequest>& q) : queue(q) {
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
        for(size_t i = 0; i < workers.size(); i++){
            queue.push(InferenceRequest{-1, 0, std::chrono::steady_clock::now()});
        }
        for (auto& worker : workers) {
            if (worker.joinable()) worker.join();
        }
    }

    int get_active_workers() const { return active_workers.load(); }
    int get_total_processed() const { return total_processed.load(); }   
    int get_total_batches() const { return total_batches.load(); }
    long long get_total_latency_ms() const { return total_latency_ms.load(); }

    Percentiles get_percentiles() {
        std::lock_guard<std::mutex> lock(samples_mutex);
        if (latency_samples.empty()) return {0.0, 0.0};

        std::vector<double> sorted_samples = latency_samples;
        std::sort(sorted_samples.begin(), sorted_samples.end());

        double p50 = sorted_samples[(size_t)(sorted_samples.size() * 0.50)];
        double p99 = sorted_samples[(size_t)(sorted_samples.size() * 0.99)];

        return {p50, p99};
    }

    TelemetrySnapshot get_and_reset_telemetry() {
        return {
            telemetry.max_queue_depth.exchange(0, std::memory_order_relaxed),
            telemetry.tasks_processed.exchange(0, std::memory_order_relaxed),
            telemetry.active_worker_count_peak.exchange(0, std::memory_order_relaxed),
            telemetry.worker_active_time_ns.exchange(0, std::memory_order_relaxed)
        };
    }
};