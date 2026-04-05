import grpc
import inference_pb2
import inference_pb2_grpc
import concurrent.futures

def send_request(client_id):
    channel = grpc.insecure_channel('localhost:50051')
    stub = inference_pb2_grpc.InferenceEngineStub(channel)
    # Sending a massive 10,000 token payload
    request = inference_pb2.InferenceRequest(tokens=list(range(1, 10001)))
    
    try:
        stub.RunInference(request)
        print(f"Client {client_id}: Success")
    except grpc.RpcError as e:
        print(f"Client {client_id}: {e.details()}")

# Fire 10 clients at the exact same time
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    for i in range(10):
        executor.submit(send_request, i)