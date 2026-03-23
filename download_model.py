from huggingface_hub import hf_hub_download

print("Downloading 67MB AI Model...")
hf_hub_download(
    repo_id="Xenova/distilbert-base-uncased-finetuned-sst-2-english",
    filename="onnx/model_quantized.onnx",
    local_dir="."
)
print("Download complete! The file is in the 'onnx' folder.")
