"""Axolotl-Q on Qollab: the experiment runs live here, then the quantum brain goes to IonQ.

1. LIVE EXPERIMENT (seed 0): train a classical and a quantum brain, damage the
   integrate module at 20/30/40%, and recover with regeneration (B, D) or
   without (A, C). Same code and seed as seed 0 of the 10-seed study; the
   browser's math differs slightly, so the numbers are an independent run.
2. IONQ TEST: run the quantum module of the healthy, 40%-damaged, and
   regenerated (D) brain on the selected backend in ONE job (Qollab allows one
   job per run), and compare its decisions with the exact simulation.
3. Print the 10-seed study summary (results/summary.json), which ran on the
   author's laptop (26 minutes on 10 cores), too long for one Qollab run.

This effort is supported by Qollab & IonQ.
"""
import json
import os
import time
import warnings
import numpy as np

warnings.filterwarnings("ignore")

from qiskit import QuantumCircuit, transpile
from models import Agent, circuit_unitary, quantum_layer_fast, N_IN, N_H, N_ACT
from pond import Pond, train_ponds, unseen_ponds, evaluate
from learn import es_train, damage, regenerate, plain_retrain

LIVE_SEED = 0          # same as seed 0 of the 10-seed study
BASE_GENS, RECOVER_GENS, LR = 100, 25, 0.1
DAMAGE_LEVELS = (0.2, 0.3, 0.4)

SHOTS = 4096
ACTIONS = ["up", "down", "left", "right"]


def find_backend():
    import builtins
    import __main__
    for place in (__main__, builtins):
        b = getattr(place, "backend", None)
        if b is not None and hasattr(b, "run"):
            return b
    return None


def live_experiment():
    """Seed-0 run of the 2x2 experiment, exactly as in experiment.py (without the generations stage)."""
    tr, un = train_ponds(), unseen_ponds()
    seed = LIVE_SEED
    bases = {}
    print("=== Part 1: live experiment on Qollab (seed 0) ===")
    for quantum in (False, True):
        t = time.time()
        a = Agent(quantum, seed)
        es_train(a, tr, BASE_GENS, np.random.default_rng(seed * 100 + quantum), lr=LR)
        bases[quantum] = a
        print(f"trained {'quantum' if quantum else 'classical'} brain: training {evaluate(a.policy(), tr):5.2f}  "
              f"unseen {evaluate(a.policy(), un):5.2f}  ({a.n_params} parameters, {time.time() - t:.0f}s)")
    rows, keep = [], {}
    for name, (quantum, regen) in {"A": (False, False), "B": (False, True), "C": (True, False), "D": (True, True)}.items():
        for frac in DAMAGE_LEVELS:
            rng = np.random.default_rng(seed * 1000 + int(frac * 100))  # same damage for with / without regrowth
            ag = bases[quantum].copy()
            pre = evaluate(ag.policy(), tr)
            damage(ag, "integrate", frac, rng)
            post = evaluate(ag.policy(), tr)
            if quantum and frac == 0.4:
                keep["damaged"] = ag.copy()
            if regen:
                regenerate(ag, tr, RECOVER_GENS, rng)
            else:
                plain_retrain(ag, tr, RECOVER_GENS, rng)
            final = evaluate(ag.policy(), tr)
            rows.append((name, frac, pre, post, final))
            if name == "D" and frac == 0.4:
                keep["regenerated"] = ag
    keep["healthy"] = bases[True]
    print("\nmodel  damage  healthy  after damage  after recovery")
    label = {"A": "A classical", "B": "B classical+regen", "C": "C quantum", "D": "D Axolotl-Q"}
    for name, frac, pre, post, final in rows:
        print(f"{label[name]:18s} {int(frac*100):3d}%   {pre:6.2f}     {post:6.2f}        {final:6.2f}")
    print("(one seed only: results vary by seed; see the 10-seed summary at the end)")
    return {k: keep[k] for k in ("healthy", "damaged", "regenerated")}


def sense(agent, o):
    th = agent.theta * agent.mask
    return np.tanh(o @ th[agent.slices["sense"]].reshape(N_IN, N_H))


def decide(agent, z):
    th = agent.theta * agent.mask
    return z @ th[agent.slices["decide"]].reshape(N_H, N_ACT)


def add_block(qc, offset, h, theta, n=N_H, layers=2):
    """Same circuit as models.circuit_unitary; simulator qubit q -> Qiskit qubit offset+(n-1-q)."""
    Q = lambda q: offset + (n - 1 - q)
    for q in range(n):
        qc.ry(np.pi * np.tanh(h[q]), Q(q))
    k = 0
    for _ in range(layers):
        for q in range(n):
            qc.ry(theta[k], Q(q)); k += 1
            qc.rz(theta[k], Q(q)); k += 1
        for q in range(n):
            qc.cx(Q(q), Q((q + 1) % n))


def z_from_counts(counts, offset, n=N_H):
    total = sum(counts.values())
    z = np.zeros(n)
    for key, c in counts.items():
        bits = key.replace(" ", "")[::-1]  # bits[i] = Qiskit qubit i
        for q in range(n):
            z[q] += (1 - 2 * int(bits[offset + (n - 1 - q)])) * c
    return z / total


def probe_observations(k):
    """Real situations from the Pond: the first k observations of training pond 0."""
    p = Pond(seed=0)
    obs = [p.reset()]
    for a in (3, 1, 3, 1):
        o, _, done = p.step(a)
        obs.append(o)
    return obs[:k]


def main():
    brains = live_experiment()
    print("\n=== Part 2: quantum brain on the selected backend ===")

    backend = find_backend()
    where = getattr(backend, "name", "local statevector") if backend is not None else "local statevector"
    n_probes = 2 if "ionq" in str(where).lower() else 1  # 24 qubits on IonQ, 12 locally
    probes = probe_observations(n_probes)

    blocks = []  # (brain name, probe index, sense output, masked integrate angles)
    for name, a in brains.items():
        th = (a.theta * a.mask)[a.slices["integrate"]]
        for i, o in enumerate(probes):
            blocks.append((name, i, sense(a, o), th))

    qc = QuantumCircuit(N_H * len(blocks))
    for b, (_, _, h, th) in enumerate(blocks):
        add_block(qc, N_H * b, h, th)
    qc.measure_all()
    print(f"\nOne circuit: {len(blocks)} brain-probes side by side, {qc.num_qubits} qubits, {SHOTS} shots")

    if backend is not None:
        counts = backend.run(transpile(qc, backend, optimization_level=1), shots=SHOTS).result().get_counts()
    else:
        from qiskit.quantum_info import Statevector
        sv = Statevector(qc.remove_final_measurements(inplace=False))
        sv.seed(7)
        counts = sv.sample_counts(SHOTS)
    print(f"sampled on: {where}\n")

    agree, errs = 0, []
    for b, (name, i, h, th) in enumerate(blocks):
        a = brains[name]
        z_hw = z_from_counts(counts, N_H * b)
        z_ex = quantum_layer_fast(h, circuit_unitary(th))
        act_hw = int(np.argmax(decide(a, z_hw)))
        act_ex = int(np.argmax(decide(a, z_ex)))
        agree += act_hw == act_ex
        errs.append(np.abs(z_hw - z_ex).mean())
        print(f"{name:12s} probe {i}: <Z> measured {np.round(z_hw, 2)}  exact {np.round(z_ex, 2)}  "
              f"decision {ACTIONS[act_hw]:5s} vs {ACTIONS[act_ex]:5s} {'match' if act_hw == act_ex else 'DIFFERENT'}")
    print(f"\nDecisions that match the exact simulation: {agree}/{len(blocks)}")
    print(f"Average output error from hardware noise and sampling: {np.mean(errs):.3f}")

    path = next((p for p in ("results/summary.json", "summary.json") if os.path.exists(p)), None)
    if path:
        s = json.load(open(path))
        print(f"\n=== Part 3: 10-seed study ({s['n_seeds']} seeds, run on the author's laptop; code in run_study.py) ===")
        for t in s["tests"]:
            shown = f"{t['rel']*100:+6.1f}%" if t.get("unit", "%") == "%" else f"{t['rel']:+6.2f} (abs)"
            print(f"{t['name']:44s} {t['x']:7.3f} vs {t['y']:7.3f}  better by {shown:>13s}  p={t['p']:.3f}  "
                  f"{'SUPPORTED' if t['supported'] else 'not supported'}")


main()
