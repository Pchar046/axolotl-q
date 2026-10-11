# Axolotl-Q

**Can a quantum AI brain regenerate after damage, like an axolotl?**

This effort is supported by Qollab & IonQ.

**Interactive app:** [pchar046.github.io/axolotl-q](https://pchar046.github.io/axolotl-q/). Damage a brain and watch it regenerate, cell by cell.

## The idea

When part of an axolotl's brain is damaged, stem-like cells (ependymoglial cells) multiply, move to the injury, and become new neurons that reconnect, and the animal recovers its behavior. Axolotl-Q borrows that idea for AI: damage part of an agent's "brain," let it **regrow and reconnect** the missing part, then let it **replicate** across generations. A **quantum layer** sits at the center of the brain, and we test whether it changes any of this.

The biology is inspiration only. We measure *functional* recovery in AI; we make no claim that it matches the animal.

## The experiment

Four models, identical in size (72 trainable parameters) and training budget:

| | No regeneration | Regeneration |
|---|---|---|
| **Classical** | A | B |
| **Quantum** | C | D (Axolotl-Q) |

- **Brain:** sense → integrate → decide. In C and D, the **integrate** module is a 4-qubit circuit (two layers of RY/RZ rotations with a CNOT ring) that matches Qiskit exactly. In A and B, it's a classical layer with the same number of parameters (16).
- **Task: "Pond."** A 6×6 grid world: collect food, avoid hazards. 8 training ponds, plus 8 **unseen** ponds with new layouts, more hazards, and moving hazards (the "smarter" test).
- **Learning:** evolution strategies, applied the same way to every model.
- **Damage:** remove 20%, 30%, or 40% of the integrate module (classical weights, or quantum gate angles).
- **Regeneration (B, D):** detect the injured module, regrow its missing parts from a small "stem" start, then retrain only that module. A and C get the same retraining budget, but the lost parts stay lost.
- **Replication:** 5 generations, 4 mutated offspring each, keep the best (selected only on training ponds).
- **Proof rule, set before running** (`PLAN.md`): 10 seeds; a hypothesis counts as supported only if p < 0.05 (Mann–Whitney) **and** the effect is at least 10%.

## Results (10-seed study, run on my own computer)

**None of the four hypotheses passed the pre-set bar.** That's an honest result, and the patterns are still informative.

| Hypothesis | What happened | Verdict |
|---|---|---|
| **H1:** quantum handles damage more gracefully | **The opposite trend.** Right after damage, the quantum brain lost about 2–3× more than the classical one (at 30% damage: 108% vs. 40% of its healthy score). | Not supported (the trend went the other way) |
| **H2:** regeneration helps recovery | **Strongest positive trend, for the quantum brain.** At 40% damage, regenerating quantum brains got back to a median of **100%** of their healthy score, vs. **69%** without regeneration, and they recovered **2–3× faster** (8 vs. 19 training rounds). Classical brains recovered fully either way. | Not supported (p ≈ 0.13–0.29) |
| **H3:** generations get smarter on unseen ponds | No consistent gain. Generation-to-generation results were noisy for every model. | Not supported |
| **H4:** quantum is more efficient | With equal parameters, the classical brain scored higher (training 2.50 vs. 1.59; unseen 1.13 vs. 0.41). | Not supported |

**The story in plain words:** in this small setup, the quantum brain was **more fragile** than the classical one, but **axolotl-style regeneration helped the quantum brain the most**, closing the gap it left. With 10 seeds those trends aren't strong enough to call proven. More seeds and bigger circuits are the natural next step.

Full numbers: `analysis_output.txt` (pre-registered tests), `seed_0.json` to `seed_9.json` (raw data), `summary.json`, and `run_info.txt` (the computer that ran it). The 10-seed study ran on my own laptop (Windows 11, AMD, 10 CPU cores, 26 minutes); `main.py` reruns seed 0 live on Qollab.

## What runs on Qollab

`main.py` runs in three parts:

1. **Live experiment (seed 0).** It trains a classical and a quantum brain, damages them at 20/30/40%, and recovers each one with or without regeneration (models A–D). It uses the same code and random seed as seed 0 of the 10-seed study. Qollab runs Python in the browser, where the math differs very slightly, and over 100 training rounds that grows into a different brain. So this is an independent live run, not an exact copy of seed 0.
2. **IonQ test.** The quantum module of the **healthy, 40%-damaged, and regenerated** brain from part 1 runs on the selected backend in **one job** (Qollab allows one job per run). On IonQ Aria 1 (Qollab's noise-model simulator) that's 6 brain-and-situation pairs side by side, 24 qubits. It checks whether the noisy outputs lead to the same decisions as the exact simulation.
3. **10-seed summary.** It prints the results of the full study.

**Where each result comes from:** the brains are trained by simulation (on Qollab in part 1). The trained quantum brains are tested on the IonQ Aria 1 noise model in part 2. The **10-seed study ran on my own laptop** (10 CPU cores, 26 minutes), because it's too long for one Qollab run. Its code is included (`experiment.py`), so anyone can rerun it.

### Live run on Qollab (IonQ Aria 1)

Full output: `qollab_ionq_aria1_run1.txt`.

| Quantum brain, recovered score (healthy = 2.09) | 20% damage | 30% damage | 40% damage |
|---|---|---|---|
| C, no regeneration | 2.05 | 1.02 | **1.25** |
| D, Axolotl-Q (regeneration) | **2.10** | **2.00** | 1.07 |

- At 30% damage, regeneration nearly **doubled** recovery (2.00 vs. 1.02). At 40%, the plain retrain did slightly better. One seed can go either way, which is why the 10-seed study matters.
- The classical brain (2.40 healthy) recovered fully in every case, matching the 10-seed study.
- **IonQ Aria 1 test:** the healthy, damaged, and regenerated quantum brains made the **same decision as the exact simulation in 6 of 6 cases**, with an average output error of 0.026 from noise and sampling. The trained quantum brains hold up under IonQ's noise model.

## How to run

**On Qollab:** click **Run Project** and choose a backend. IonQ Aria 1 runs 24 qubits; the built-in simulator runs a lighter 12-qubit version. Training happens first, so nothing prints for several minutes.

**Full experiment, on your own computer** (26 minutes on 10 cores):
```
pip install -r requirements-full.txt
python run_study.py         # all 10 seeds on every CPU core, then the tests
python main.py              # live seed-0 run + quantum test
```

## Files

| File | What it does |
|---|---|
| `main.py` | Qollab entry point: live seed-0 experiment, then the IonQ test of healthy, damaged, and regenerated brains |
| `pond.py` | The Pond world: training and unseen ponds |
| `models.py` | Modular agent; the quantum layer simulated in NumPy (checked against Qiskit) |
| `learn.py` | Evolution-strategies training, damage, regeneration |
| `experiment.py` | The full 4-model, 10-seed experiment |
| `run_study.py` | Runs the whole 10-seed study on all CPU cores and records which computer ran it |
| `analyze.py` | Tests the hypotheses with the pre-set rule |
| `index.html` | Interactive results app (GitHub Pages) |
| `PLAN.md` | The research plan, written before running |
| `seed_*.json`, `summary.json`, `analysis_output.txt`, `run_info.txt` | 10-seed study results from my laptop |
| `qollab_ionq_aria1_run1.txt` | Full output of the live Qollab run on IonQ Aria 1 |

## Honest limits

- **Small:** a 4-qubit module, a 6×6 world, and 10 seeds. Real effects could appear at larger sizes, or disappear.
- **Simulated training:** the brains were trained in simulation. The trained quantum module was tested on Qollab's IonQ Aria 1 noise model, not the physical QPU.
- **Where it ran:** the 10-seed study ran on my laptop. Only seed 0 is rerun live on Qollab.
- **Unseen-pond scores are noisy:** moving hazards draw new random moves each time a brain is tested, so unseen scores depend on test order. Run each seed in its own process (as `run_study.py` does) to reproduce them exactly. Training-pond results (H1, H2) aren't affected.
- **"Damage" for quantum** means zeroing gate angles, one choice among several possible.
- **Regeneration detection is simple:** it finds the module with the most missing parts. A real system would need to detect damage from behavior alone.
- **Replication is bounded** (fixed offspring, inside the simulation) and showed no clear benefit here.

## Next steps

- 30+ seeds, to confirm or rule out the regeneration trend.
- Bigger circuits (6–8 qubits) and harder ponds.
- Detect damage from behavior only, the way an animal would.
- Use real hardware damage (a noisy or dead qubit) instead of simulated damage.
- Test and train on IonQ QPU hardware.

## Related work

- Mordvintsev et al., "Growing Neural Cellular Automata" (Distill, 2020): networks that regrow damaged patterns.
- Neuroevolution: evolving networks through mutation and selection.
- Quantum reinforcement learning surveys (2022, 2025).
- Albantakis, Prentner, Durham, "Computing the Integrated Information of a Quantum Mechanism" (Entropy, 2023).

## Disclosure

**Libraries:** Qiskit, NumPy, SciPy (analysis only), Qiskit Aer (local testing only).
**Templates:** none. Built from scratch during the hackathon window.
**AI assistance:** built with help from Claude (Anthropic). At my direction, Claude researched prior work, wrote the code. I ran the 10 seed study on my own computer. I came up with the axolotl-inspired idea and the four-model design, directed the research, and made the decisions.

## License

MIT License. See `LICENSE`.

Author: Patrick Charles (Qollab: Charpat160)
