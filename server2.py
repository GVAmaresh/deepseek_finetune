import os
import sys
import subprocess
import torch
import wandb
from huggingface_hub import login
from unsloth import FastLanguageModel

hf_token = "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"
wb_token = "6cc6bf1ce7c5fe75ceefa32067d562ec7338a963"

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",  
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

def install_dependencies():
    print("🔄 Installing dependencies...")
    subprocess.run([
        "pip", "install", "--upgrade", "unsloth", "kaggle", "wandb"
    ], check=True)

def setup_huggingface():
    print("🔑 Logging into Hugging Face...")
    login(token=hf_token)

def setup_wandb():
    print("🚀 Logging into WandB...")
    wandb.login(key=wb_token)
    wandb.init(
        project="Fine-tune-DeepSeek-R1-Distill-Llama-8B on Medical COT Dataset",
        job_type="training"
    )

def clone_kaggle_repo():
    print("📂 Cloning Kaggle repo (if needed)...")
    if not os.path.exists("docker-python"):
        subprocess.run(["git", "clone", "https://github.com/Kaggle/docker-python.git"], check=True)
        sys.path.append("./docker-python/patches")

def check_cuda():
    print("🔍 Checking CUDA availability...")

    if torch.cuda.is_available():
        print(f"✅ CUDA Available: {torch.cuda.is_available()}")
        print(f"🚀 GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"🔥 PyTorch CUDA Version: {torch.version.cuda}")
    else:
        print("❌ No GPU detected! Check your WSL setup or driver installation.")

def load_model():
    print("📥 Loading Mistral-NeMo-Minitron-8B-Base model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="google/gemma-2-2b",
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
        device_map="auto", 
        llm_int8_enable_fp32_cpu_offload=True,
        token=hf_token,
    )
    print("✅ Model loaded successfully!")
    return model, tokenizer

def main():
    install_dependencies()
    setup_huggingface()
    setup_wandb()
    clone_kaggle_repo()
    check_cuda()
    
    # Load the model
    model, tokenizer = load_model()

    print("🎉 Setup Complete! Ready for training.")

if __name__ == "__main__":
    main()
