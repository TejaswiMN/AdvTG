import os
import torch
from trl import PPOConfig

# Environment variables
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# os.environ['HTTP_PROXY'] = '127.0.0.1:7890'
# os.environ['HTTPS_PROXY'] = '127.0.0.1:7890'
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model paths
model_name_or_path = "/content/drive/MyDrive/MLSecResearch/code/model/ppo_model/Text/100"

# Features dictionary
features_dict = {"Image": "../model/imgae_model_configs.pkl", "Text": "../model/model_configs.pkl"}

# PPO Configuration
def create_ppo_config():
    return PPOConfig(
        is_peft_model=True,
        model_name=model_name_or_path,
        learning_rate=1.41e-5,
        batch_size=4,
        mini_batch_size=1,
        gradient_accumulation_steps=4,
        use_score_scaling=True,  # scaling
        use_score_norm=True,  # normalization
        score_clip=1,
        log_with="wandb"
    )

# Generation configurations
sent_kwargs = {
    "return_all_scores": True,
    "function_to_apply": "none",
    "batch_size": 4  # This should match config.forward_batch_size
}

# Generation parameters
output_min_length = 128
output_max_length = 256
max_length = 512
query_max_length = 128

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": None  # Will be set after tokenizer is loaded
}

# Logging configuration
columns_to_log = ['text', 'instruction', 'input', 'output', 'response'] 
