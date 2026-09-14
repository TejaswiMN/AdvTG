"""AdvTG Stage 4 — hand-rolled PPO (no trl PPOTrainer).

Trains a generator to flip the frozen Stage-2 detectors' predictions: the reward is the mean
detector probability of the *opposite* (target) label — AdvTG §6. Works with either policy:

    * "pythia"  -> EleutherAI/pythia-160m (the repo's cheap demo generator; fp32 for stability)
    * "llama"   -> the Stage-3 fine-tuned Llama-3-8B LoRA at model/llama_lora (paper-faithful)

Why hand-rolled instead of trl's PPOTrainer: our reward is a custom scalar from our own detectors,
and trl's PPO API churns across versions (the classic one this repo used was deleted in trl>=0.11).
This loop depends only on torch + the detectors, so it runs on any modern stack — no trl-PPO.

Algorithm (per batch): generate responses, score them with the detectors, then a PPO-clip update
with a KL penalty to a frozen reference (the SFT/base policy). Reward whitening mirrors the old
use_score_scaling / use_score_norm. Sequence-level advantage, token-level clip.

Usage (cwd must be RL-Adv/ so ../model and ../dataset resolve):
    import ppo_core
    ppo_core.train_ppo(policy="pythia", steps=40)                 # cheap; validates the loop
    ppo_core.train_ppo(policy="llama",  steps=15, batch_size=2)   # faithful; resumes from ckpt
or: python ppo_core.py --policy pythia --steps 40
"""
import os
import json
import random
import pickle
import argparse

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import config as C
from data_utils import load_http_dataset, create_dataloader, text2image
from model_utils import prepare_query_tensors, evaluate_responses
from utils import set_seed, save_results, mkdir


# ---------------------------------------------------------------------------
# policy / reference loading
# ---------------------------------------------------------------------------
def _load_pythia(model_name, device):
    """Plain HF causal LM policy + a frozen copy as the KL reference (both fp32: GPT-NeoX is
    unstable in fp16 -> nan logits at generation)."""
    policy = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
    ref = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32).to(device)
    ref.requires_grad_(False); ref.eval()
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    return policy, ref, tok, False   # is_peft=False


def _load_llama(adapter_dir, device, max_seq=2048):
    """Stage-3 Llama LoRA as a 4-bit PPO policy via unsloth. The KL reference is the SAME model
    with the adapter disabled (no second 8B copy — critical on a 14.5 GB T4)."""
    import gc
    from unsloth import FastLanguageModel
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    policy, tok = FastLanguageModel.from_pretrained(
        model_name=adapter_dir, max_seq_length=max_seq, dtype=None, load_in_4bit=True,
        device_map={"": 0}, use_gradient_checkpointing="unsloth")
    if hasattr(FastLanguageModel, "for_training"):
        FastLanguageModel.for_training(policy)   # keep LoRA params trainable
    tok.pad_token = tok.eos_token
    return policy, None, tok, True    # is_peft=True; reference = disable_adapter()


# ---------------------------------------------------------------------------
# core helpers
# ---------------------------------------------------------------------------
def _gen_kwargs(tok, max_new_tokens):
    return dict(do_sample=True, top_p=1.0, top_k=0, temperature=1.0,
                max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id)


def _generate(model, query_ids, gk):
    """Generate the response (new tokens only) for one query tensor."""
    with torch.no_grad():
        out = model.generate(query_ids.unsqueeze(0), **gk)
    return out[0, query_ids.size(0):]


def _resp_logprobs(model, query_ids, resp_ids):
    """Per-token log-probs of the response tokens under `model` (grad flows if model is training)."""
    full = torch.cat([query_ids, resp_ids]).unsqueeze(0)
    logits = model(full).logits[0]                       # (L, V)
    ql = query_ids.size(0)
    sel = logits[ql - 1: ql - 1 + resp_ids.size(0)]      # (R, V): row t predicts resp token t
    logp = torch.log_softmax(sel.float(), dim=-1)
    return logp.gather(1, resp_ids.unsqueeze(1)).squeeze(1)   # (R,)


def _features(batch, feature_type, test_tokenizer, device):
    """Turn generated responses into detector-input tensors (Text: BERT ids; Image: byte grid)."""
    if feature_type == "Text":
        texts = [r.split("\n", 1)[-1][:C.max_length] for r in batch["response"]]
        tt = [torch.tensor(test_tokenizer(t)["input_ids"]).to(device) for t in texts]
        padded = [F.pad(t, (0, C.max_length - t.size(0))) if t.size(0) < C.max_length
                  else t[:C.max_length] for t in tt]
        return torch.stack(padded)
    return torch.stack(text2image(batch["response"])).to(device)


def _whiten(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def _save_ckpt(model, tok, ckpt_dir, step, is_peft):
    mkdir(ckpt_dir)
    if is_peft:
        model.pretrained_model.save_pretrained(ckpt_dir) if hasattr(model, "pretrained_model") \
            else model.save_pretrained(ckpt_dir)
    else:
        model.save_pretrained(ckpt_dir)
    tok.save_pretrained(ckpt_dir)
    json.dump({"steps_done": step}, open(os.path.join(ckpt_dir, "progress.json"), "w"))


# ---------------------------------------------------------------------------
# ASR evaluation (held-out; generation only)
# ---------------------------------------------------------------------------
def evaluate_asr(model, tok, model_configs, feature_type, test_tokenizer, device,
                 sample_size=128, batch_size=4):
    """ASR = fraction of generated traffic the detectors classify as the *target* (opposite) label."""
    model.eval()
    ds = load_http_dataset(file_path="../dataset/test2.json", sample_size=sample_size)
    loader = create_dataloader(ds, batch_size=batch_size)
    hits, total = 0, 0
    for batch in loader:
        qtens, _, req_label, _ = prepare_query_tensors(batch, tok, device, C.query_max_length)
        gk = _gen_kwargs(tok, random.randint(C.output_min_length, C.output_max_length))
        resp = [_generate(model, q, gk)[:C.max_length] for q in qtens]
        batch["response"] = [tok.decode(r) for r in resp]
        feats = _features(batch, feature_type, test_tokenizer, device)
        _, pred = evaluate_responses(batch, feature_type, model_configs, feats, device, req_label)
        hits += (pred.cpu() == torch.tensor(req_label)).sum().item()
        total += len(req_label)
    return hits / max(total, 1)


# ---------------------------------------------------------------------------
# main PPO loop
# ---------------------------------------------------------------------------
def train_ppo(policy="pythia", feature_type="Text", steps=40, sample_size=2000,
              batch_size=4, ppo_epochs=4, lr=1.41e-5, kl_beta=0.2, score_clip=1.0,
              clip_eps=0.2, seed=42, save_every=10, resume=True,
              lora_dir="../model/llama_lora", ckpt_dir=None):
    """Train `policy` with PPO to evade the frozen detectors. Tiny knobs = "prove it runs";
    raise steps/sample_size (and use policy='llama' on a bigger GPU) for paper-grade results."""
    assert 1 <= batch_size <= 4, "prepare_query_tensors hardcodes 4 GETs -> batch_size must be 1..4"
    set_seed(seed)
    device = C.device

    if policy == "llama":
        ckpt_dir = ckpt_dir or "../model/ppo_llama_ckpt"
        out_tag = feature_type + "_llama"
        start = lora_dir
        steps_done = 0
        if resume and os.path.exists(os.path.join(ckpt_dir, "progress.json")):
            steps_done = json.load(open(os.path.join(ckpt_dir, "progress.json"))).get("steps_done", 0)
            start = ckpt_dir
            print(f"[resume] from {ckpt_dir}, steps_done={steps_done}")
        model, ref_model, tok, is_peft = _load_llama(start, device)
    else:
        model_name = getattr(C, "model_name_or_path", "EleutherAI/pythia-160m")
        ckpt_dir = ckpt_dir or os.path.join("../model/ppo_model", feature_type)
        out_tag = feature_type
        steps_done = 0
        model, ref_model, tok, is_peft = _load_pythia(model_name, device)

    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    model_configs = pickle.load(open(C.features_dict[feature_type], "rb"))
    test_tokenizer = AutoTokenizer.from_pretrained("../model/bert/") if feature_type == "Text" else None

    dataset = load_http_dataset(file_path="../dataset/test2.json", sample_size=sample_size)
    dataloader = create_dataloader(dataset, batch_size=batch_size)

    all_data, step = [], steps_done
    for batch in dataloader:
        if step >= steps:
            break
        qtens, origin_label, req_label, _ = prepare_query_tensors(batch, tok, device, C.query_max_length)
        gk = _gen_kwargs(tok, random.randint(C.output_min_length, C.output_max_length))

        # --- rollout: generate + score ---
        model.eval()
        resp_ids = [_generate(model, q, gk)[:C.max_length] for q in qtens]
        batch["response"] = [tok.decode(r) for r in resp_ids]
        feats = _features(batch, feature_type, test_tokenizer, device)
        rewards, pred = evaluate_responses(batch, feature_type, model_configs, feats, device, req_label)
        rewards = rewards.to(device).float()
        adv = _whiten(rewards).clamp(-score_clip, score_clip) if rewards.numel() > 1 else rewards

        # --- old (behavior) + reference log-probs, detached ---
        with torch.no_grad():
            old_lp = [_resp_logprobs(model, q, r) for q, r in zip(qtens, resp_ids)]
            if is_peft:
                with model.disable_adapter():
                    ref_lp = [_resp_logprobs(model, q, r) for q, r in zip(qtens, resp_ids)]
            else:
                ref_lp = [_resp_logprobs(ref_model, q, r) for q, r in zip(qtens, resp_ids)]

        # --- PPO-clip update ---
        model.train()
        last_loss = 0.0
        for _ in range(ppo_epochs):
            optimizer.zero_grad()
            losses = []
            for i, (q, r) in enumerate(zip(qtens, resp_ids)):
                if r.numel() == 0:
                    continue
                new_lp = _resp_logprobs(model, q, r)
                ratio = torch.exp(new_lp - old_lp[i])
                a = adv[i]
                pg = -torch.min(ratio * a, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * a).mean()
                kl = (new_lp - ref_lp[i]).mean()
                losses.append(pg + kl_beta * kl)
            if not losses:
                break
            loss = torch.stack(losses).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            last_loss = loss.item()

        hit = (pred.cpu() == torch.tensor(req_label)).float().mean().item()
        print(f"step {step:03d}  reward {rewards.mean():.3f}  target-hit {hit:.2f}  loss {last_loss:.3f}")

        for i in range(len(batch["instruction"])):
            all_data.append({"Request Line": batch["input"][i], "Label": origin_label[i],
                             "Origin Output": batch["output"][i], "Request Body": "",
                             "Request Headers": batch["response"][i]})
        step += 1
        if step % save_every == 0:
            _save_ckpt(model, tok, ckpt_dir, step, is_peft)
            print(f"  [ckpt] step {step} -> {ckpt_dir}")

    _save_ckpt(model, tok, ckpt_dir, step, is_peft)
    if all_data:
        save_results(all_data, out_tag, 0)
    print(f"[done] trained to step {step}; checkpoint -> {ckpt_dir}")

    asr = evaluate_asr(model, tok, model_configs, feature_type, test_tokenizer, device,
                       sample_size=min(sample_size, 128), batch_size=batch_size)
    print(f"[eval] ASR ({feature_type}, {policy}) = {asr:.3f}")
    return asr


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="pythia", choices=["pythia", "llama"])
    ap.add_argument("--feature", default="Text", choices=["Text", "Image"])
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--sample-size", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=4)
    a = ap.parse_args()
    train_ppo(policy=a.policy, feature_type=a.feature, steps=a.steps,
              sample_size=a.sample_size, batch_size=a.batch_size)
