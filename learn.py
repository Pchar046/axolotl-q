"""Learning, damage, and regeneration.

Training uses evolution strategies (ES): try small random changes to the
parameters, keep moving toward the changes that score better. It works the
same way for classical and quantum modules, which keeps the comparison fair.
"""
import numpy as np
from pond import evaluate


def es_train(agent, ponds, gens, rng, only=None, pop=12, sigma=0.15, lr=0.08):
    """Train the agent in place. `only` limits learning to one module.
    Returns the score after each generation."""
    free = agent.mask.copy()
    if only is not None:
        keep = np.zeros_like(free)
        keep[agent.slices[only]] = 1
        free *= keep
    history = []
    for _ in range(gens):
        eps = rng.normal(0, 1, (pop, agent.theta.size)) * free
        scores = np.empty(2 * pop)
        for i in range(pop):
            scores[2 * i] = evaluate(agent.policy(agent.theta + sigma * eps[i]), ponds)
            scores[2 * i + 1] = evaluate(agent.policy(agent.theta - sigma * eps[i]), ponds)
        # rank-based scoring: robust to noisy rewards (same for every model)
        ranks = np.empty(2 * pop)
        ranks[np.argsort(scores)] = np.linspace(-0.5, 0.5, 2 * pop)
        diff = ranks[0::2] - ranks[1::2]
        agent.theta += lr / (pop * sigma) * (diff @ eps)
        history.append(evaluate(agent.policy(), ponds))
    return history


def damage(agent, module, frac, rng):
    """Remove a fraction of a module's parameters (weights or quantum gate angles)."""
    sl = agent.slices[module]
    idx = np.arange(sl.start, sl.stop)
    hit = rng.choice(idx, size=int(round(frac * idx.size)), replace=False)
    agent.mask[hit] = 0
    return hit


def detect_damage(agent):
    """Find the module with the most missing parameters (the injured area)."""
    lost = {k: 1 - agent.mask[s].mean() for k, s in agent.slices.items()}
    return max(lost, key=lost.get)


def regenerate(agent, ponds, budget, rng):
    """Axolotl-style repair: regrow missing parts from a stem template,
    then retrain only the injured module while the rest stays fixed."""
    module = detect_damage(agent)
    sl = agent.slices[module]
    missing = np.where(agent.mask[sl] == 0)[0] + sl.start
    agent.theta[missing] = rng.normal(0, 0.1, missing.size)  # new "cells" start small
    agent.mask[missing] = 1
    return module, es_train(agent, ponds, budget, rng, only=module)


def plain_retrain(agent, ponds, budget, rng):
    """No regeneration: the lost parts stay lost; the rest keeps learning."""
    return es_train(agent, ponds, budget, rng)
