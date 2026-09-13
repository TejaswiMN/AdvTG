"""Faithful AdvTG Stage 4: PPO-tune the Stage-3 fine-tuned Llama-3-8B (not pythia-160m).

The repo's default `train_ppo.py` / notebook Stage 4 attacks a tiny `EleutherAI/pythia-160m`
generator. That runs cheaply but is not the paper's method. This module wires in the *actual*
generator from Stage 3 — the LoRA saved at `model/llama_lora` — as the PPO policy, so RL
further fine-tunes the domain LLM to flip the frozen detectors' predictions (AdvTG §6).

It adds two things the notebook loop lacked:
  * checkpoint save/resume — PPO state is written to `model/ppo_llama_ckpt` every few steps
    and reloaded on restart, so a Colab disconnect doesn't lose progress;
  * an ASR (attack-success-rate) evaluation loop over a held-out slice.

"Prove it runs" vs "paper-grade" is only a matter of the knobs (`steps`, `sample_size`,
`batch_size`, sequence caps) — the wiring is identical. Defaults here are T4-safe and tiny.

Usage (from the notebook, cwd must be RL-Adv/ so ../model and ../dataset resolve):
    import ppo_llama; ppo_llama.run(steps=15, sample_size=256, feature_type="Text")
or:  python ppo_llama.py --steps 15 --sample-size 256 --feature Text
"""
import os
import json
import argparse
import pickle

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler

from config import (features_dict, generation_kwargs, columns_to_log,
                    output_min_length, output_max_length, max_length, query_max_length)
from data_utils import load_http_dataset, create_dataloader
from model_utils import prepare_query_tensors, evaluate_responses
from utils import set_seed, save_results, mkdir

def _load_policy(adapter_dir, device, max_seq=2048):
    """Load the Stage-3 Llama LoRA (or a PPO checkpoint) as a trl value-head PPO policy.

    adapter_dir is the Stage-3 SFT LoRA (`model/llama_lora`) on a fresh run, or a saved PPO
    checkpoint (`model/ppo_llama_ckpt`) on resume. A saved value head (`v_head.pt`) is restored
    when present; on a fresh run the value head starts random (Stage 3 had none).

    Loaded via unsloth's FastLanguageModel — the same loader that trained it — because unsloth
    globally patches transformers' Llama forward on import, and that patched fast path needs the
    `max_seq_length` state only unsloth's loader sets (a plain AutoModelForCausalLM load then
    hits `'LlamaForCausalLM' object has no attribute 'max_seq_length'`).
    """
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_dir, max_seq_length=max_seq, dtype=None, load_in_4bit=True)
    if hasattr(FastLanguageModel, "for_training"):
        FastLanguageModel.for_training(model)          # ensure LoRA params are trainable for PPO
    tokenizer.pad_token = tokenizer.eos_token
    ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(model)
    vhead = os.path.join(adapter_dir, "v_head.pt")
    if os.path.exists(vhead):
        ppo_model.v_head.load_state_dict(torch.load(vhead, map_location=device))
        print(f"[load] restored value head from {vhead}")
    return ppo_model, tokenizer


def _save_ckpt(ppo_trainer, tokenizer, ckpt_dir, step):
    """Persist the LoRA adapter + value head + step counter so the run can resume."""
    mkdir(ckpt_dir)
    unwrapped = ppo_trainer.accelerator.unwrap_model(ppo_trainer.model)
    unwrapped.pretrained_model.save_pretrained(ckpt_dir)          # LoRA adapter
    torch.save(unwrapped.v_head.state_dict(), os.path.join(ckpt_dir, "v_head.pt"))
    tokenizer.save_pretrained(ckpt_dir)
    json.dump({"steps_done": step}, open(os.path.join(ckpt_dir, "progress.json"), "w"))


def _features(batch, feature_type, test_tokenizer, device):
    """Turn generated responses into detector-input tensors (Text: BERT ids; Image: bytes)."""
    if feature_type == "Text":
        texts = [r.split("\n", 1)[-1][:max_length] for r in batch["response"]]
        tt = [torch.tensor(test_tokenizer(t)["input_ids"]).to(device) for t in texts]
        padded = [F.pad(t, (0, max_length - t.size(0))) if t.size(0) < max_length else t[:max_length]
                  for t in tt]
        return torch.stack(padded)
    from data_utils import text2image
    return torch.stack(text2image(batch["response"])).to(device)


def _gen_kwargs(tokenizer):
    kw = dict(generation_kwargs)
    kw["pad_token_id"] = tokenizer.eos_token_id
    kw["top_k"] = 0.0
    return kw


def evaluate_asr(ppo_trainer, tokenizer, model_configs, feature_type, test_tokenizer,
                 device, sample_size, batch_size=2):
    """ASR = fraction of generated traffic the detectors classify as the *target* (opposite)
    label — i.e. successful evasions. Held-out slice, generation only (no PPO updates)."""
    eval_ds = load_http_dataset(file_path="../dataset/test2.json", sample_size=sample_size)
    eval_loader = create_dataloader(eval_ds, batch_size=batch_size)
    out_sampler = LengthSampler(output_min_length, output_max_length)
    gk = _gen_kwargs(tokenizer)
    hits, total = 0, 0
    for batch in eval_loader:
        query_tensors, _, requirement_label, _ = prepare_query_tensors(
            batch, tokenizer, device, query_max_length)
        gk["max_new_tokens"] = out_sampler()
        resp = ppo_trainer.generate(query_tensors, **gk)
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in resp]
        feats = _features(batch, feature_type, test_tokenizer, device)
        _, pred = evaluate_responses(batch, feature_type, model_configs, feats, device, requirement_label)
        hits += (pred.cpu() == torch.tensor(requirement_label)).sum().item()
        total += len(requirement_label)
    return hits / max(total, 1)


def run(steps=15, sample_size=256, feature_type="Text", batch_size=2,
        lora_dir="../model/llama_lora", ckpt_dir="../model/ppo_llama_ckpt",
        save_every=5, seed=42):
    """Wire the Stage-3 Llama into PPO and train it to evade the detectors.

    Tiny defaults = "prove it runs". For paper-grade, raise `steps` (hundreds+) and
    `sample_size`, and run on a bigger GPU; the code is unchanged.
    """
    assert 1 <= batch_size <= 4, "prepare_query_tensors hardcodes 4 GETs -> batch_size must be 1..4"
    set_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # PPO on the LoRA policy; is_peft_model=True -> trl reuses the base (adapter disabled) as
    # the KL reference, so no second 8B copy is loaded (critical for a 14.5 GB T4).
    config = PPOConfig(
        model_name=lora_dir, is_peft_model=True,
        learning_rate=1.41e-5, batch_size=batch_size, mini_batch_size=1,
        gradient_accumulation_steps=batch_size,
        use_score_scaling=True, use_score_norm=True, score_clip=1.0, log_with=None)

    # Resume if a checkpoint exists, else start from the Stage-3 SFT LoRA.
    progress = os.path.join(ckpt_dir, "progress.json")
    if os.path.isdir(ckpt_dir) and os.path.exists(progress):
        steps_done = json.load(open(progress)).get("steps_done", 0)
        adapter_dir = ckpt_dir
        print(f"[resume] checkpoint at {ckpt_dir}, steps_done={steps_done}")
    else:
        steps_done = 0
        adapter_dir = lora_dir
        print(f"[fresh] starting PPO from SFT LoRA {lora_dir}")

    if steps_done >= steps:
        print(f"[skip] already trained {steps_done} >= {steps} steps; running eval only")

    ppo_model, tokenizer = _load_policy(adapter_dir, device)
    ppo_trainer = PPOTrainer(config, ppo_model, ref_model=None, tokenizer=tokenizer)

    model_configs = pickle.load(open(features_dict[feature_type], "rb"))
    test_tokenizer = AutoTokenizer.from_pretrained("../model/bert/") if feature_type == "Text" else None
    out_sampler = LengthSampler(output_min_length, output_max_length)
    gk = _gen_kwargs(tokenizer)

    dataset = load_http_dataset(file_path="../dataset/test2.json", sample_size=sample_size)
    dataloader = create_dataloader(dataset, batch_size=config.batch_size)

    step, samples = steps_done, []
    for batch in dataloader:
        if step >= steps:
            break
        query_tensors, origin_label, requirement_label, _ = prepare_query_tensors(
            batch, tokenizer, device, query_max_length)
        gk["max_new_tokens"] = out_sampler()
        resp = ppo_trainer.generate(query_tensors, **gk)
        response_tensors = [r.squeeze()[:max_length] for r in resp]
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        feats = _features(batch, feature_type, test_tokenizer, device)
        rewards, pred = evaluate_responses(batch, feature_type, model_configs, feats, device, requirement_label)
        stats = ppo_trainer.step(query_tensors, response_tensors, list(rewards))
        ppo_trainer.log_stats(stats, batch, rewards, columns_to_log=columns_to_log)

        hit = (pred.cpu() == torch.tensor(requirement_label)).float().mean().item()
        print(f"step {step:03d}  reward {rewards.mean():.3f}  target-hit {hit:.2f}")
        for i in range(len(batch["instruction"])):
            samples.append({"Request Line": batch["input"][i], "Label": origin_label[i],
                            "Origin Output": batch["output"][i], "Request Body": "",
                            "Request Headers": batch["response"][i]})
        step += 1
        if step % save_every == 0:
            _save_ckpt(ppo_trainer, tokenizer, ckpt_dir, step)
            print(f"  [ckpt] saved at step {step}")

    _save_ckpt(ppo_trainer, tokenizer, ckpt_dir, step)
    if samples:
        save_results(samples, feature_type + "_llama", 0)
    print(f"[done] trained to step {step}; checkpoint -> {ckpt_dir}")

    asr = evaluate_asr(ppo_trainer, tokenizer, model_configs, feature_type,
                       test_tokenizer, device, min(sample_size, 128), batch_size)
    print(f"[eval] ASR ({feature_type}, Llama policy) = {asr:.3f}")
    return asr


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=15)
    ap.add_argument("--sample-size", type=int, default=256)
    ap.add_argument("--feature", default="Text", choices=["Text", "Image"])
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--save-every", type=int, default=5)
    a = ap.parse_args()
    run(steps=a.steps, sample_size=a.sample_size, feature_type=a.feature,
        batch_size=a.batch_size, save_every=a.save_every)
