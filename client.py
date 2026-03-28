import grpc
import inference_pb2
import inference_pb2_grpc

def run():
    print("Attempting to connect to C++ Engine on port 50051...")
    channel = grpc.insecure_channel('localhost:50051')
    
    stub = inference_pb2_grpc.InferenceEngineStub(channel)
    
    request = inference_pb2.InferenceRequest(
        tokens=list(range(1, 7001))
    )
    
    print("Firing payload...")
    
    
    try:
        response = stub.RunInference(request)
        print("Sent successfully! Look at your C++ terminal!")
    except grpc.RpcError as e:
        print(f"Network error: {e.details()}")

if __name__ == '__main__':
    run()