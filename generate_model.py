import torch
import torch.nn as nn

# 1. Create a microscopic neural network
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Takes 1 input, gives 1 output
        self.linear = nn.Linear(1, 1) 

    def forward(self, x):
        # Convert incoming int64 tokens to floats so the math works
        x = x.to(torch.float32)
        return self.linear(x)

# 2. Instantiate it
model = DummyModel()


dummy_input = torch.tensor([[101], [4054], [8976]], dtype=torch.int64)


torch.onnx.export(
    model, 
    dummy_input, 
    "build/Debug/model.onnx", # Save it exactly where the .exe will look for it
    input_names=['input'], 
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}} # Tell ONNX the batch size can change
)

print("model.onnx successfully generated in build/Debug/")