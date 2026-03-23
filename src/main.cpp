#include <iostream>
#include <thread>
#include <grpcpp/grpcpp.h>
#include "inference.grpc.pb.h" 
#include "crow.h"
#include "simple_queue.hpp"
#include "thread_pool.hpp"

SimpleQueue<InferenceRequest> order_queue;
class InferenceServiceImpl final : public inference::InferenceEngine::Service {
    grpc::Status RunInference(grpc::ServerContext* context, const inference::InferenceRequest* request, inference::InferenceResponse* reply) override {
        int tokens = request->tokens_size();
        std::cout << "Received a request with " << tokens << " tokens." << std::endl;
        for (int i = 0; i < tokens; i++) {
            int token = request->tokens(i);
            
           order_queue.push({(int64_t)token, 1, std::chrono::steady_clock::now()});
        }
        
        reply->add_output_tokens(200);

        return grpc::Status::OK;

        
        
        
    }
};



int main() {
    unsigned int n = std::thread::hardware_concurrency();
    std::cout << "Starting " << n << " worker threads." << std::endl;

    ThreadPool pool(n, order_queue);
    std::thread dashboard_thread([&pool]() {
        crow::SimpleApp app;

        CROW_ROUTE(app, "/metrics")([&]() {
            crow::json::wvalue x;
            x["queue_depth"] = order_queue.get_queue_depth();
            x["active_workers"] = pool.get_active_workers();
            x["total_processed"] = pool.get_total_processed();
            x["total_batches"] = pool.get_total_batches();
            x["total_latency_ms"] = pool.get_total_latency_ms();
            return x;
        });

        std::cout << "Dashboard listening on http://localhost:8080/metrics" << std::endl;
        app.port(8080).multithreaded().run();
    });
    dashboard_thread.detach();
    std::string server_address("0.0.0.0:50051");
    InferenceServiceImpl service;
    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    server->Wait();
    return 0;
}