#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <chrono>
#include <atomic>

template <typename T>
class SimpleQueue {
private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cond_;
    std::atomic<int> current_queue_depth{0};

public:
    void push(T item) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(item);
        current_queue_depth++;
        // Use notify_all so all workers can check if the batch threshold is met
        cond_.notify_all();
    }

    T pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        while(queue_.empty()) {
            cond_.wait(lock);
        }
        T item = queue_.front();
        queue_.pop();
        current_queue_depth--;
        return item;
    }

    std::vector<T> pop_batch(size_t batch_size, int timeout_ms) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        
        cond_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this, batch_size]() {
            return queue_.size() >= batch_size;
        });

        std::vector<T> batch;
        while(!queue_.empty() && batch.size() < batch_size) {
            batch.push_back(queue_.front());
            queue_.pop();
            current_queue_depth--;
        }

        return batch;
    }

    int get_queue_depth() const {
        return current_queue_depth.load();
    }
};