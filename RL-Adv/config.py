import os
import torch
# NOTE: no `from trl import PPOConfig` — Stage 4 PPO is hand-rolled in ppo_core.py and does not
# depend on trl's PPO API (which changes across versions). trl is used only for Stage 3 SFT.

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
model_name_or_path = "EleutherAI/pythia-160m"

# Features dictionary
features_dict = {"Image": "../model/imgae_model_configs.pkl", "Text": "../model/model_configs.pkl"}

# PPO hyperparameters (read/overridable as ppo_core.train_ppo(...) args; kept here as the
# tuning surface). The hand-rolled PPO in ppo_core.py uses these defaults.
PPO_LEARNING_RATE = 1.41e-5
PPO_BATCH_SIZE    = 4
PPO_EPOCHS        = 4       # PPO-clip inner epochs per batch
PPO_KL_BETA       = 0.2     # KL penalty vs the frozen reference (SFT/base policy)
PPO_SCORE_CLIP    = 1.0     # clamp on whitened advantage (was use_score_scaling/norm + score_clip)
PPO_CLIP_EPS      = 0.2     # PPO ratio clip epsilon

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
