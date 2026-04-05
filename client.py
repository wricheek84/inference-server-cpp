import grpc
import inference_pb2
import inference_pb2_grpc

def run():
    print("Attempting to connect to C++ Engine on port 50051...")
    channel = grpc.insecure_channel('localhost:50051')
    
    stub = inference_pb2_grpc.InferenceEngineStub(channel)
    
    # We now have to pass 'model_id' so the C++ server knows 
    # to use the DistilBERT session we named "classifier".
    request = inference_pb2.InferenceRequest(
        model_id="classifier",  # <--- THIS IS THE FIX
        tokens=list(range(1, 7001))
    )
    
    print(f"Firing payload of {len(request.tokens)} tokens for model 'classifier'...")
    
    try:
        response = stub.RunInference(request)
        print("Sent successfully! Check your C++ terminal for the worker logs.")
    except grpc.RpcError as e:
        print(f"Network error: {e.code()} - {e.details()}")

if __name__ == '__main__':
    run()