"""Stage 4 PPO entry point — thin CLI over ppo_core.train_ppo (hand-rolled PPO, no trl-PPO).

    python train_ppo.py --policy pythia --steps 40            # cheap demo generator
    python train_ppo.py --policy llama  --steps 15 --batch-size 2   # Stage-3 Llama (paper-faithful)
"""
import argparse

from ppo_core import train_ppo


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="pythia", choices=["pythia", "llama"])
    ap.add_argument("--feature", default="Text", choices=["Text", "Image"])
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--sample-size", type=int, default=2000)
    ap.add_argument("--batch-size", type=int, default=4)
    a = ap.parse_args()
    train_ppo(policy=a.policy, feature_type=a.feature, steps=a.steps,
              sample_size=a.sample_size, batch_size=a.batch_size)


if __name__ == "__main__":
    main()
