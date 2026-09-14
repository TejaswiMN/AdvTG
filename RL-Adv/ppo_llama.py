"""Deprecated shim. The faithful Stage-4 Llama PPO now lives in ppo_core.py as policy="llama"
(a single hand-rolled loop serves both pythia and Llama, with no trl-PPO dependency).

Kept so existing calls `import ppo_llama; ppo_llama.run(...)` still work.
"""
from ppo_core import train_ppo


def run(steps=15, sample_size=256, feature_type="Text", batch_size=2,
        lora_dir="../model/llama_lora", ckpt_dir=None, save_every=5, seed=42):
    return train_ppo(policy="llama", feature_type=feature_type, steps=steps,
                     sample_size=sample_size, batch_size=batch_size,
                     lora_dir=lora_dir, ckpt_dir=ckpt_dir, save_every=save_every, seed=seed)
