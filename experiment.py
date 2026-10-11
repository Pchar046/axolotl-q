"""Axolotl-Q full experiment for one seed.

For each seed:
 1. Train a classical base and a quantum base (identical budgets).
 2. Damage the integrate module at 20/30/40%.
    A = classical, no regeneration   B = classical + regeneration
    C = quantum,   no regeneration   D = quantum + regeneration (Axolotl-Q)
 3. From the 30%-damage recovered models, run 5 generations of bounded
    replication (4 offspring each, mutate -> retrain -> keep the best).
Results are written to results/seed_<n>.json as soon as the seed finishes.

Usage: python experiment.py <seed> [<seed> ...]
"""
import json
import os
import sys
import time
import numpy as np
from models import Agent
from pond import train_ponds, unseen_ponds, evaluate
from learn import es_train, damage, regenerate, plain_retrain

BASE_GENS = 100
RECOVER_GENS = 25
DAMAGE_LEVELS = (0.2, 0.3, 0.4)
N_GENERATIONS = 5
N_OFFSPRING = 4
OFFSPRING_GENS = 8
MUTATION = 0.05
LR = 0.1

TRAIN, UNSEEN = train_ponds(), unseen_ponds()
MODELS = {"A": (False, False), "B": (False, True), "C": (True, False), "D": (True, True)}


def recover(agent, regen, rng):
    if regen:
        _, hist = regenerate(agent, TRAIN, RECOVER_GENS, rng)
    else:
        hist = plain_retrain(agent, TRAIN, RECOVER_GENS, rng)
    return hist


def evolve(parent, rng):
    """Bounded replication: N_OFFSPRING mutated copies per generation, keep the best."""
    record = [{"gen": 0, "train": evaluate(parent.policy(), TRAIN),
               "unseen": evaluate(parent.policy(), UNSEEN)}]
    for g in range(1, N_GENERATIONS + 1):
        kids = []
        for _ in range(N_OFFSPRING):
            kid = parent.copy()
            kid.theta += rng.normal(0, MUTATION, kid.theta.size) * kid.mask
            es_train(kid, TRAIN, OFFSPRING_GENS, rng, lr=LR)
            kids.append((evaluate(kid.policy(), TRAIN), kid))
        kids.sort(key=lambda t: t[0], reverse=True)
        parent = kids[0][1]  # selected only on training ponds
        record.append({"gen": g, "train": kids[0][0], "unseen": evaluate(parent.policy(), UNSEEN)})
    return record


def run_seed(seed):
    out = {"seed": seed, "config": {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, tuple))}}
    bases = {}
    for quantum in (False, True):
        rng = np.random.default_rng(seed * 100 + quantum)
        a = Agent(quantum, seed)
        t = time.time()
        hist = es_train(a, TRAIN, BASE_GENS, rng, lr=LR)
        bases[quantum] = a
        out[f"base_{'q' if quantum else 'c'}"] = {
            "curve": hist, "train": hist[-1], "unseen": evaluate(a.policy(), UNSEEN),
            "params": a.n_params, "seconds": time.time() - t}
    for name, (quantum, regen) in MODELS.items():
        base = bases[quantum]
        res = {"damage": {}}
        for frac in DAMAGE_LEVELS:
            rng = np.random.default_rng(seed * 1000 + int(frac * 100))  # same damage pattern for regen / no-regen
            ag = base.copy()
            pre_t, pre_u = evaluate(ag.policy(), TRAIN), evaluate(ag.policy(), UNSEEN)
            damage(ag, "integrate", frac, rng)
            post_t, post_u = evaluate(ag.policy(), TRAIN), evaluate(ag.policy(), UNSEEN)
            t = time.time()
            hist = recover(ag, regen, rng)
            res["damage"][str(frac)] = {
                "pre_train": pre_t, "pre_unseen": pre_u, "post_train": post_t, "post_unseen": post_u,
                "curve": hist, "final_train": hist[-1], "final_unseen": evaluate(ag.policy(), UNSEEN),
                "params_after": ag.n_params, "seconds": time.time() - t}
            if frac == 0.3:
                survivor = ag
        t = time.time()
        res["evolution"] = evolve(survivor, np.random.default_rng(seed * 7 + len(name)))
        res["evolution_seconds"] = time.time() - t
        out[name] = res
    os.makedirs("results", exist_ok=True)
    with open(f"results/seed_{seed}.json", "w") as f:
        json.dump(out, f)
    print(f"seed {seed} done", flush=True)


if __name__ == "__main__":
    for s in sys.argv[1:]:
        run_seed(int(s))
