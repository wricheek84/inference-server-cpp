#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>

template <typename T>
class SimpleQueue {
private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cond_;

public:
    // Pushes an item into the queue
    void push(T item) {
        std::lock_guard<std::mutex> lock(mutex_);
        queue_.push(item);
        cond_.notify_one();
       
    }

    // Pops an item and returns it
    T pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        while(queue_.empty()) {
            cond_.wait(lock);
        }
        T item = queue_.front();
        queue_.pop();
        
        return item;
    }
};