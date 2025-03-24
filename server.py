import os
import sys
import subprocess
from huggingface_hub import login
import torch
import wandb

import torch
print("CUDA Available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")
print("PyTorch CUDA Version:", torch.version.cuda)
print("Torch Version:", torch.__version__)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
print("CUDA Available:", torch.cuda.is_available())

from unsloth import FastLanguageModel
print("Unsloth imported successfully!")


hf_token="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"

def install_dependencies():
    subprocess.run(["pip", "install", "unsloth"], check=True)
    subprocess.run(["pip", "install", "--force-reinstall", "--no-cache-dir", "--no-deps", "git+https://github.com/unslothai/unsloth.git"], check=True)
    subprocess.run(["pip", "install", "kaggle", "wandb"], check=True)

def setup_kaggle():
    os.environ["KAGGLE_CONFIG_DIR"] = "./"
    # subprocess.run(["kaggle", "datasets", "list"], check=True)

def setup_huggingface(hf_token: str):
    login(token=hf_token)

def setup_wandb(wb_token: str):
    wandb.login(key=wb_token)
    run = wandb.init(
        project="Fine-tune-DeepSeek-R1-Distill-Llama-8B on Medical COT Dataset",
        job_type="training",
        anonymous="allow"
    )

def clone_kaggle_repo():
    subprocess.run(["git", "clone", "https://github.com/Kaggle/docker-python.git"], check=True)
    sys.path.append("./docker-python/patches")

def check_cuda():
    print("CUDA Available:", torch.cuda.is_available())
    print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")
    print("PyTorch CUDA Version:", torch.version.cuda)
    
def importing_model():

    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    max_seq_length = 2048
    dtype = None
    load_in_4bit = True


    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/DeepSeek-R1-Distill-Llama-8B",
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        token = hf_token,
    )

def main():
    install_dependencies()
    setup_kaggle()
    
    # hf_token = "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"
    setup_huggingface(hf_token)
    
    wb_token = "6cc6bf1ce7c5fe75ceefa32067d562ec7338a963"
    setup_wandb(wb_token)
    
    clone_kaggle_repo()
    
    print("✅ Setup Complete! Kaggle, Hugging Face, and WandB are ready to use.")
    check_cuda()
    importing_model()
    print("Model imported successfully")
    


if __name__ == "__main__":
    main()
