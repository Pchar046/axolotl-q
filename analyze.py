"""Test the four hypotheses on whatever seeds have finished.

Rule set in PLAN.md: a hypothesis is supported only if the Mann-Whitney
test gives p < 0.05 AND the difference is at least 10%.
"""
import glob
import json
import numpy as np
from scipy.stats import mannwhitneyu

R = [json.load(open(f)) for f in sorted(glob.glob("results/seed_*.json"))]
LEVELS = ["0.2", "0.3", "0.4"]


def col(model, fn):
    return np.array([fn(r[model]) for r in R])


def drop_frac(d):
    """Share of performance lost right after damage (relative to the healthy score)."""
    return (d["pre_train"] - d["post_train"]) / max(abs(d["pre_train"]), 1e-9)


def recovered_frac(d):
    """Final score as a share of the healthy score."""
    return d["final_train"] / max(abs(d["pre_train"]), 1e-9)


def time_to_90(d):
    target = 0.9 * d["pre_train"]
    for i, v in enumerate(d["curve"]):
        if v >= target:
            return i + 1
    return len(d["curve"]) + 1


def gain(m):
    e = m["evolution"]
    return e[-1]["unseen"] - e[0]["unseen"]


def test(name, x, y, higher_is_better=True):
    """Compare x (claimed better) against y."""
    alt = "greater" if higher_is_better else "less"
    p = mannwhitneyu(x, y, alternative=alt).pvalue if len(x) >= 3 else float("nan")
    mx, my = np.median(x), np.median(y)
    sign = 1 if higher_is_better else -1
    if abs(my) < 0.05:  # comparison value near zero: use the absolute difference instead of a percentage
        rel, unit = (mx - my) * sign, "abs"
    else:
        rel, unit = (mx - my) / abs(my) * sign, "%"
    ok = p < 0.05 and rel >= 0.10
    shown = f"{rel*100:+6.1f}%" if unit == "%" else f"{rel:+6.2f} (abs)"
    print(f"  {name:44s} median {mx:7.3f} vs {my:7.3f}  better by {shown:>13s}  p={p:.3f}  {'SUPPORTED' if ok else 'not supported'}")
    return {"name": name, "x": float(mx), "y": float(my), "rel": float(rel), "unit": unit, "p": float(p), "supported": bool(ok)}


def main():
    n = len(R)
    print(f"Seeds finished: {n}\n")
    out = {"n_seeds": n, "tests": [], "summary": {}}

    print("Base agents (healthy):")
    for k in ("base_c", "base_q"):
        tr = np.array([r[k]["train"] for r in R]); un = np.array([r[k]["unseen"] for r in R])
        sec = np.mean([r[k]["seconds"] for r in R]); par = R[0][k]["params"]
        print(f"  {k}: train {np.median(tr):.2f}  unseen {np.median(un):.2f}  params {par}  avg {sec:.0f}s")
        out["summary"][k] = {"train": float(np.median(tr)), "unseen": float(np.median(un)), "params": par, "seconds": float(sec)}

    print("\nH1 graceful damage: quantum loses less right after damage (C vs A, lower is better)")
    for L in LEVELS:
        out["tests"].append(test(f"H1 drop at {L}", col("C", lambda m: drop_frac(m["damage"][L])),
                                 col("A", lambda m: drop_frac(m["damage"][L])), higher_is_better=False))

    print("\nH2 regeneration helps: share of healthy score regained")
    for L in LEVELS:
        out["tests"].append(test(f"H2 classical B vs A at {L}", col("B", lambda m: recovered_frac(m["damage"][L])),
                                 col("A", lambda m: recovered_frac(m["damage"][L]))))
        out["tests"].append(test(f"H2 quantum D vs C at {L}", col("D", lambda m: recovered_frac(m["damage"][L])),
                                 col("C", lambda m: recovered_frac(m["damage"][L]))))
    print("  recovery time (generations to 90% of healthy, lower is better):")
    for L in LEVELS:
        out["tests"].append(test(f"H2 time D vs C at {L}", col("D", lambda m: time_to_90(m["damage"][L])),
                                 col("C", lambda m: time_to_90(m["damage"][L])), higher_is_better=False))

    print("\nH3 generations get smarter: unseen-pond gain over 5 generations, D vs each")
    for other in ("A", "B", "C"):
        out["tests"].append(test(f"H3 gain D vs {other}", col("D", gain), col(other, gain)))

    print("\nH4 efficiency: same parameter count, compare healthy score (quantum vs classical)")
    out["tests"].append(test("H4 train score Q vs C (equal params)",
                             np.array([r["base_q"]["train"] for r in R]), np.array([r["base_c"]["train"] for r in R])))
    out["tests"].append(test("H4 unseen score Q vs C (equal params)",
                             np.array([r["base_q"]["unseen"] for r in R]), np.array([r["base_c"]["unseen"] for r in R])))

    # tables for the write-up and the results page
    tab = {}
    for m in "ABCD":
        tab[m] = {L: {"drop": float(np.median(col(m, lambda x: drop_frac(x["damage"][L])))),
                      "recovered": float(np.median(col(m, lambda x: recovered_frac(x["damage"][L])))),
                      "t90": float(np.median(col(m, lambda x: time_to_90(x["damage"][L]))))} for L in LEVELS}
        ev = np.array([[g["unseen"] for g in r[m]["evolution"]] for r in R])
        tab[m]["evolution_unseen_median"] = np.median(ev, axis=0).tolist()
        tab[m]["gain"] = float(np.median(col(m, gain)))
    out["table"] = tab
    print("\nMedian table:")
    for m in "ABCD":
        print(f"  {m}: " + "  ".join(f"{L}: drop {tab[m][L]['drop']*100:5.1f}% rec {tab[m][L]['recovered']*100:5.1f}%" for L in LEVELS)
              + f"  | gen gain {tab[m]['gain']:+.2f}")
    json.dump(out, open("results/summary.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
