#include <iostream>
#include <thread>
#include <vector>
#include <mutex>
#include <set>
#include <grpcpp/grpcpp.h>
#include "inference.grpc.pb.h" 
#include "crow.h"
#include "simple_queue.hpp"
#include "thread_pool.hpp"

SimpleQueue<InferenceRequest> order_queue;

class InferenceServiceImpl final : public inference::InferenceEngine::Service {
    grpc::Status RunInference(grpc::ServerContext* context, const inference::InferenceRequest* request, inference::InferenceResponse* reply) override {
        int tokens = request->tokens_size();
        

        if (order_queue.get_queue_depth() > 500) {
            return grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED, "Server is overloaded");
        }

        std::cout << "Received request for model: " << request->model_id() << " with " << tokens << " tokens." << std::endl;

        for (int i = 0; i < tokens; i++) {
            int token = request->tokens(i);
            
            
            order_queue.push({
                request->model_id(), 
                (int64_t)token, 
                1, 
                std::chrono::steady_clock::now()
            });
        }

        
        reply->add_output_tokens(200); 
        return grpc::Status::OK;
    }
};

int main() {
    
    unsigned int n = std::thread::hardware_concurrency();
    std::cout << "Starting " << n << " worker threads." << std::endl;

    
    std::vector<std::pair<std::string, std::wstring>> models = {
        {"classifier", L"C:\\Users\\wrich\\inference-server-cpp\\onnx\\model_quantized.onnx"}
    };


    ThreadPool pool(n, order_queue, models);


    std::thread dashboard_thread([&pool]() {
        crow::SimpleApp app;
        std::mutex mtx;
        std::set<crow::websocket::connection*> users;

        CROW_WEBSOCKET_ROUTE(app, "/ws")
            .onopen([&](crow::websocket::connection& conn) {
                std::lock_guard<std::mutex> lock(mtx);
                users.insert(&conn);
            })
            .onclose([&](crow::websocket::connection& conn, const std::string& reason, uint16_t code) {
                std::lock_guard<std::mutex> lock(mtx);
                users.erase(&conn);
            });

        std::thread broadcaster([&]() {
            while (true) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                
                auto p = pool.get_percentiles();
                auto snap = pool.get_and_reset_telemetry();
                
                crow::json::wvalue x;
                x["queue_peak"] = snap.queue_peak;
                x["tasks_processed"] = snap.tasks_processed;
                x["worker_peak"] = snap.worker_peak;
                x["worker_active_time_ns"] = snap.worker_active_time_ns;
                x["p50_latency"] = p.p50;
                x["p99_latency"] = p.p99;
                x["total_batches"] = pool.get_total_batches();
                x["latest_prediction"] = pool.get_latest_prediction();
                x["latest_confidence"] = (double)pool.get_latest_confidence();

                std::string msg = x.dump();
                
                std::lock_guard<std::mutex> lock(mtx);
                for (auto u : users) {
                    u->send_text(msg);
                }
            }
        });
        broadcaster.detach();

        std::cout << "WebSocket telemetry active on ws://localhost:8080/ws" << std::endl;
        app.port(8080).multithreaded().run();
    });
    dashboard_thread.detach();

    // --- gRPC Server Setup ---
    std::string server_address("0.0.0.0:50051");
    InferenceServiceImpl service;
    grpc::ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);
    
    std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
    std::cout << "gRPC Inference Server listening on " << server_address << std::endl;
    server->Wait();

    return 0;
}