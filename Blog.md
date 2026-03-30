---
title: Squeezing 2,240 TPS out of a 2019 Laptop: Building a C++ Inference Engine
published: false
description: How I built a high-performance DistilBERT server using C++, gRPC, and ONNX Runtime.
tags: cpp, machinelearning, backend, performance
---

In current times, for running AI models, the NVIDIA H100 is a top-tier choice. It has almost 16,000+ CUDA cores, a huge 80 GB High Bandwidth Memory (HBM), and a raw computing power of around 10¹⁵ operations/sec or 1000+ TFLOPS.

I built my project on a 2019 HP 15 series with an AMD Ryzen 5 3500U processor with Radeon Vega 8. It has 8 threads with 8 GB DDR4 RAM, out of which 5.92 GB is usable.  

The aim was not to try and beat industry standards, but to squeeze the very best out of limited hardware by relying on batching, threading, and system design.

At their core, AI models are massive chains of linear algebra:

Matrix × Vector = Results

High-end GPUs use specialized cores to do this in parallel; I had to make my CPU threads do it as efficiently as possible given the constraints. In high-end systems, the GPU performs the math, ultra-fast VRAM stores the model, and thousands of operations run in parallel.

I attempted to work within these constraints by batching requests, using thread pools efficiently without causing race conditions, and optimizing CPU usage.

Large models need almost 16 GB, 32 GB, or even 80 GB just to load. In cases of insufficient RAM, the computer starts using the SSD, which is much slower than RAM.  

So the focus was simple: keep everything in RAM and push the CPU as hard as possible.

Most AI uses Python as its primary language, and even though it has its positives, it uses a Global Interpreter Lock (GIL), which limits true parallel execution for CPU-bound tasks. Combined with garbage collection pauses, this can impact performance in high-throughput systems.

On my laptop, C++ allowed for maximum efficiency, letting me squeeze as much computation as possible from my 4 Ryzen cores.

---

## Architecture Overview

ARCHITECTURE DIAGRAM  
![System Architecture](./assets/architecture_diagram.png)

To make this work efficiently, I structured the system into multiple layers:
- Communication (gRPC)
- Orchestration (Thread Pool + Batching)
- Inference (ONNX Runtime)

---

## Communication Layer (gRPC)

In a standard AI setup, a REST API sending JSON data is more likely. While JSON is easy to debug, it’s "heavy." Every request requires the CPU to parse text strings into usable data — cycles I simply couldn't afford to waste on a 4-core Ryzen processor.

I chose gRPC because it treats a remote server method as if it were a local object. More importantly, it uses Protocol Buffers, a binary serialization format. Instead of "reading" sentences, my server receives a compact binary stream.

Here is the "contract" I defined. I used `repeated int32` so the server receives pre-processed IDs, avoiding a heavy tokenizer on the backend:

Protocol Buffers

```
message InferenceRequest {
  repeated int32 tokens = 1; 
}

message InferenceResponse {
  repeated int32 output_tokens = 1;
}
```

This means the moment a packet hits the server, it’s ready for the math engine.

---

## Thread Pool (Orchestration Layer)

Simple apps often use a "thread-per-request" model.

On a Ryzen 3500U, that breaks pretty quickly. Under heavy load (say 7000 requests), most of the CPU ends up managing threads instead of doing actual computation.

So I used a fixed thread pool with 8 workers — matching the 8 logical threads available.

```cpp
unsigned int n = std::thread::hardware_concurrency();
std::cout << "Starting " << n << " worker threads." << std::endl;

ThreadPool pool(n, order_queue);
```

Efficient wait and batching logic:

```cpp
std::vector<InferenceRequest> requests = queue.pop_batch(32, 25);
```

The 32-token limit: modern CPUs are most efficient when doing vectorized math. Processing 32 tokens together allows ONNX Runtime to leverage SIMD instructions instead of running 1 token 32 times.

The 25 ms wait: under low load, I didn’t want requests waiting forever just to fill a batch. This caps latency — after 25 ms, whatever is available gets processed.

There’s a trade-off here: better throughput at the cost of slightly higher latency under load.

---

## Non-Busy Waiting (Resource Efficiency)

Inside that `pop_batch` call is a `std::condition_variable`.

If the queue is empty, workers don’t spin and waste CPU cycles. They sleep.

The moment gRPC pushes new work, one of them wakes up instantly.

So the system stays quiet at idle and jumps straight to 100% CPU when a burst hits.

---

## Inference Layer (Architecture over Algorithm)

The engine itself is model-agnostic. I used a generic `Ort::Session`, so the system can run any ONNX model.

DistilBERT was just a reference.

```cpp
session = std::make_unique<Ort::Session>(
    env,
    L"C:\\Users\\wrich\\inference-server-cpp\\onnx\\model_quantized.onnx",
    session_options
);
```

Choosing C++ wasn't just about language preference; it was about removing the 'Middleman' overhead found in  AI frameworks. Most Python AI libraries are built on top of C++ backends. Data is typically wrapped in Python objects (PyObject), which then need to be converted into native types before being passed to the underlying C++ engine. The results are then converted back into Python objects.

This abstraction is convenient, but it introduces overhead. Given my hardware constraints, I chose to bypass this layer entirely and work directly in C++, using native types (int64_t) and interacting directly with ONNX Runtime’s shared libraries (.dll / .so).

---

## INT8 Quantization

A standard FP32 model is heavy (~260 MB). I used an INT8-quantized version (~67 MB).

This helped in two ways:
- fits comfortably in RAM
- integer ops are faster on CPU

More importantly, it avoids spilling into SSD memory, which would kill performance.

There is a small accuracy drop, but it was a trade-off I had to make.

---

## Avoiding Oversubscription

ONNX Runtime tries to use all CPU cores by default.

But I already had 8 worker threads. Letting ONNX spawn more threads would just create contention.

```cpp
ThreadPool(int num_threads, SimpleQueue<InferenceRequest>& q) : queue(q) {
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
}
```

This forces ONNX Runtime to use a single thread per session.

That way, my thread pool stays in control, and the CPU spends time on computation instead of context switching.

This is what gave me stable ~100% CPU usage under load.

---

## Telemetry Challenge: “Ghost Metrics”

One unexpected problem: the system was too fast.

Even after firing thousands of tokens, by the time I opened the dashboard, everything showed 0 — no load, no active workers.

The work finished faster than the UI could refresh.

---

## Why Not Mutex?

Using `std::mutex` for counters created contention.

If multiple workers finished at the same time, they’d line up waiting for the lock. That slows everything down.

---

## Atomic-Based Telemetry

So I switched to `std::atomic`.

Instead of tracking current values, I tracked peak values over a time window.

Instead of:
“How many workers are active right now?”

I tracked:
“What was the max number of active workers in the last interval?”

This made short bursts visible.

---

## Compare-And-Swap (CAS)

```cpp
int cur_peak = telemetry.active_worker_count_peak.load(std::memory_order_relaxed);
while (current_active > cur_peak && 
       !telemetry.active_worker_count_peak.compare_exchange_weak(
           cur_peak, current_active, std::memory_order_relaxed)) {}
```

Basic idea:
- read current value
- compare with new one
- update if higher
- retry if needed

`compare_exchange_weak` can fail occasionally, but inside a loop it retries immediately.

---

## Atomic Snapshot Trick

For dashboard polling:

```cpp
TelemetrySnapshot get_and_reset_telemetry() {
    return {
        telemetry.max_queue_depth.exchange(0, std::memory_order_relaxed),
        telemetry.tasks_processed.exchange(0, std::memory_order_relaxed),
        telemetry.active_worker_count_peak.exchange(0, std::memory_order_relaxed),
        telemetry.worker_active_time_ns.exchange(0, std::memory_order_relaxed)
    };
}
```

`.exchange(0)` reads and resets in one atomic step, so no updates are lost.

I used `std::memory_order_relaxed` since these are independent counters and don’t need strict ordering.

---

## Results & Benchmarks

The engine was stress tested with bursts (~7000 tokens at peak), and performance stayed stable.

![Performance Benchmark Dashboard](./assets/performance_benchmark.jpeg)

~2,240 tokens/sec  
100% CPU utilization  
No SSD paging  

---

## The “Perfect” 100% Load

Seeing CPU usage sit at 100% consistently was probably the most satisfying part.

In many systems, it hovers around 70–80% due to inefficiencies.

Here it stayed pinned.

That basically means:
- no I/O bottlenecks
- no thread contention
- CPU fully used for computation

---

## Throughput Details

This wasn’t a one-off burst.

I tested multiple patterns:
- large 7000-token bursts
- smaller rapid batches

Total processed: 38,392 tokens

The queue handled backpressure well, and batching stayed efficient.

Even with 5.92 GB RAM, nothing spilled to SSD.

At that point, the system wasn’t I/O-bound or thread-bound anymore — just compute-bound.

---

## Metrics

![Inference Engine Stress Test: 8 Active Workers and Peak Queue](./assets/latency_image.jpeg)

All 8 logical cores pinned, queue handling load without issues.

![Real-time Latency Stream during 7,000 token burst](./assets/latency_graph.jpeg)

Latency stays controlled even during ramp-up.

![Final Benchmark: Stabilized Steady-State Latency](./assets/jittery_graph.jpeg)

Flat latency under sustained load.

---

## Final Thoughts

This wasn’t about competing with GPUs.

It was about understanding where resources were being used , maximizing efficieny and extracting the best performance from limited hardware.
