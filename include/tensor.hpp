#pragma once
#include <vector>
#include <cstdint>


struct Tensor {
    std::vector<int64_t> data; 
    std::vector<int64_t> shape; 

    
    Tensor(std::vector<int>& raw_batch) {
        
        for(int token : raw_batch) {
            data.push_back(static_cast<int64_t>(token));
        }
        
        
        shape = {static_cast<int64_t>(raw_batch.size()), 1};
    }
};