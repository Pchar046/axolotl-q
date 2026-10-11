# Axolotl-Q: research plan (v1)

**Quantum-enhanced, self-regenerating AI inspired by the axolotl brain**
Patrick Charles · started October 11, 2026

---

## The question

When part of an AI's network is damaged, can it **regenerate** and recover, and can it **improve across generations**? Does a **quantum layer** change how well it does either?

## Inspiration

When part of an axolotl's brain is damaged, stem-like cells (ependymoglial cells) multiply, move to the injury, and become new neurons that reconnect, and the animal recovers its behavior. We use this as inspiration only. We test *functional* recovery in AI; we don't assume it matches biology.

## Hypotheses (written before running anything)

- **H1: Graceful damage.** A quantum layer loses less performance when damaged, because entanglement spreads information across qubits instead of storing it in one place.
- **H2: Regeneration works.** Models that regenerate recover more performance, faster, than models that only keep training.
- **H3: Generations get smarter.** Axolotl-Q (D) improves more across generations on tasks it has never seen than A, B, or C.
- **H4: Efficiency.** Quantum models reach the same score with fewer trainable parameters.

Any of these can come out false. A clear "no" is still a result.

## The four models

| | No regeneration | Regeneration |
|---|---|---|
| **No quantum** | A. Standard AI | B. Regenerative AI |
| **Quantum** | C. Quantum AI | D. Axolotl-Q |

All four share the same modular design, the same parameter budget, and the same training time.

**Modules:** sense → integrate → decide → act. In C and D, the **integrate** module is a small quantum circuit (4–6 qubits). In A and B, it's a classical layer of matched size.

## The task: "Pond"

A small grid world. The agent finds food and avoids hazards.
- **Training ponds:** a set of fixed layouts.
- **Unseen ponds:** new layouts, moving hazards, and changed rules, used only to test generalization ("smarter").

## Damage

Disable **20%, 30%, and 40%** of a chosen module:
- **Classical:** zero out a share of its weights or neurons.
- **Quantum:** remove or scramble a share of its gates.

## Regeneration (B and D only)

1. **Detect:** find the module whose outputs went dead or erratic, plus the performance drop.
2. **Regrow:** re-initialize the damaged part from a "stem" template.
3. **Reconnect:** retrain only that module, with the rest frozen, for a fixed budget.

Models A and C get the same retraining budget, but on the whole network with no targeted regrowth. That keeps the comparison fair.

## Controlled replication (bounded)

- Each generation, every model makes **4 offspring** through small mutations: weight noise, and adding or removing a neuron or gate.
- Each offspring is retrained on a fixed budget. The best one is selected.
- **5 generations.** Everything runs inside the simulation, with a fixed cap on offspring.

## What we measure

| Measure | Meaning |
|---|---|
| Learning speed | Episodes to reach a target score |
| Task score | Average reward on training ponds |
| Damage drop | Score lost right after damage |
| Recovery | Share of the original score regained |
| Recovery time | Episodes needed to regain 90% |
| Generalization | Score on unseen ponds ("smarter") |
| Generational gain | Change in unseen-pond score per generation |
| Cost | Trainable parameters, circuit runs, compute time |

## What counts as proof (set in advance)

- **10 random seeds** per model, per condition.
- Compare models with a Mann–Whitney test. Report the effect size, not just pass or fail.
- A hypothesis is **supported** only if p < 0.05 **and** the difference is meaningful (at least 10% better).
- All results get reported, including the ones that don't support the idea.

## Phases

1. **Build the pond** and the classical models A and B. Get damage and regeneration working.
2. **Add the quantum layer** for C and D on a simulator.
3. **Add generations** (replication and selection).
4. **Run the full experiment**, 10 seeds each.
5. **Test the best Axolotl-Q on IonQ** (noise model, then hardware if possible).
6. **Write it up.**

## Known risks

- **Compute time:** evolution multiplied by quantum simulation gets slow, so we start tiny.
- **Training stalls in quantum circuits** (barren plateaus): keep circuits shallow.
- **Hardware noise:** report ideal and noisy results side by side.

## Related work (starting points)

- Growing Neural Cellular Automata (2020): networks that regrow damaged images.
- Neuroevolution: evolving networks through mutation and selection.
- Quantum reinforcement learning surveys (2022, 2025).
- Octopus-inspired distributed control (Hou, Laschi et al., 2026), classical only.
