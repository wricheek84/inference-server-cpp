#include <iostream>
#include <thread>
#include <grpcpp/grpcpp.h>
#include "inference.grpc.pb.h" 

#include "simple_queue.hpp"
#include "thread_pool.hpp"

SimpleQueue<int> order_queue;
class InferenceServiceImpl final : public inference::InferenceEngine::Service {
    grpc::Status RunInference(grpc::ServerContext* context, const inference::InferenceRequest* request, inference::InferenceResponse* reply) override {
        int tokens = request->tokens_size();
        std::cout << "Received a request with " << tokens << " tokens." << std::endl;
        if(tokens>0){
            int first_token = request->tokens(0);
            std::cout << "Pushing task with first token: " << first_token << " to the queue." << std::endl;
            order_queue.push(first_token);
        }
        reply->add_output_tokens(200);

        return grpc::Status::OK;

        
        
        
    }
};



int main() {
    unsigned int n = std::thread::hardware_concurrency();
    std::cout << "Starting " << n << " worker threads." << std::endl;

    ThreadPool pool(n, order_queue);
    std::string server_address("0.0.0.0:50051");
    InferenceServiceImpl service;
    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    server->Wait();
    return 0;
}