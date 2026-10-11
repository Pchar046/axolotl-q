"""Pond: a small grid world. Find food, avoid hazards.

Observation (10 numbers): hazard or wall next to the agent in each of 4
directions, and the direction to the nearest food (dx, dy).
Actions: 0 up, 1 down, 2 left, 3 right.
"""
import numpy as np

MOVES = np.array([[0, -1], [0, 1], [-1, 0], [1, 0]])


class Pond:
    def __init__(self, seed, size=6, n_food=3, n_haz=3, moving=False, max_steps=30):
        self.rng = np.random.default_rng(seed)
        self.size, self.n_food, self.n_haz = size, n_food, n_haz
        self.moving, self.max_steps = moving, max_steps
        cells = [(x, y) for x in range(size) for y in range(size)]
        pick = self.rng.permutation(len(cells))[: 1 + n_food + n_haz]
        pts = [np.array(cells[i]) for i in pick]
        self.start = pts[0]
        self.food0 = pts[1:1 + n_food]
        self.haz0 = pts[1 + n_food:]

    def reset(self):
        self.pos = self.start.copy()
        self.food = [f.copy() for f in self.food0]
        self.haz = [h.copy() for h in self.haz0]
        self.t = 0
        return self.obs()

    def _inside(self, p):
        return 0 <= p[0] < self.size and 0 <= p[1] < self.size

    def obs(self):
        o = np.zeros(10)
        for i, m in enumerate(MOVES):
            q = self.pos + m
            o[i] = 1.0 if any((q == h).all() for h in self.haz) else 0.0
            o[4 + i] = 0.0 if self._inside(q) else 1.0
        if self.food:
            d = [np.abs(f - self.pos).sum() for f in self.food]
            f = self.food[int(np.argmin(d))]
            o[8:10] = np.sign(f - self.pos)
        return o

    def step(self, a):
        self.t += 1
        q = self.pos + MOVES[a]
        r = -0.01
        if self._inside(q):
            self.pos = q
        else:
            r -= 0.05
        if any((self.pos == h).all() for h in self.haz):
            return self.obs(), r - 1.0, True
        for i, f in enumerate(self.food):
            if (self.pos == f).all():
                self.food.pop(i)
                r += 1.0
                break
        if self.moving and self.t % 3 == 0:  # hazards drift in unseen ponds
            for h in self.haz:
                step = MOVES[self.rng.integers(4)]
                if self._inside(h + step) and not (h + step == self.pos).all():
                    h += step
        done = not self.food or self.t >= self.max_steps
        return self.obs(), r, done


def train_ponds():
    return [Pond(seed=s) for s in range(8)]


def unseen_ponds():
    """New layouts, more hazards, moving hazards: never used in training."""
    return [Pond(seed=1000 + s, n_haz=4, moving=True) for s in range(8)]


def evaluate(policy, ponds):
    """Average total reward of a greedy policy over a list of ponds."""
    total = 0.0
    for p in ponds:
        o, done, ret = p.reset(), False, 0.0
        while not done:
            o, r, done = p.step(int(np.argmax(policy(o))))
            ret += r
        total += ret
    return total / len(ponds)
