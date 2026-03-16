import grpc
import inference_pb2
import inference_pb2_grpc

def run():
    print("Attempting to connect to C++ Engine on port 50051...")
    channel = grpc.insecure_channel('localhost:50051')
    
    stub = inference_pb2_grpc.InferenceEngineStub(channel)
    
    # MATCHING YOUR CONTRACT EXACTLY:
    # Sending a list of integer tokens (e.g., [101, 456, 789])
    request = inference_pb2.InferenceRequest(
        tokens=[101, 4054, 8976] 
    )
    
    print("Firing payload...")
    
    # We will wrap this in a try-except block just in case
    try:
        response = stub.RunInference(request)
        print("Shot fired successfully! Look at your C++ terminal!")
    except grpc.RpcError as e:
        print(f"Network error: {e.details()}")

if __name__ == '__main__':
    run()