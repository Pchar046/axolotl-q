"""Run the full 10-seed Axolotl-Q study on this computer, using all CPU cores.

Usage:  python run_study.py

It clears any old results, runs seeds 0-9 in parallel, tests the hypotheses
(analyze.py), and records which computer ran it in results/run_info.txt.
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")       # one thread per worker: faster in parallel
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import contextlib
import datetime
import glob
import io
import multiprocessing as mp
import platform
import sys
import time

SEEDS = list(range(10))
HERE = os.path.dirname(os.path.abspath(__file__))


def run_one(seed):
    os.chdir(HERE)
    sys.path.insert(0, HERE)
    import experiment
    t = time.time()
    experiment.run_seed(seed)
    return seed, time.time() - t


def main():
    os.chdir(HERE)
    os.makedirs("results", exist_ok=True)
    for f in glob.glob("results/seed_*.json") + glob.glob("results/*.txt") + ["results/summary.json"]:
        if os.path.exists(f):
            os.remove(f)

    workers = min(len(SEEDS), os.cpu_count() or 1)
    start = time.time()
    print(f"Running {len(SEEDS)} seeds on {workers} CPU cores. This takes a while; each seed prints when done.", flush=True)
    with mp.Pool(workers) as pool:
        for seed, secs in pool.imap_unordered(run_one, SEEDS):
            print(f"  seed {seed} finished in {secs/60:.1f} min  (total {(time.time()-start)/60:.1f} min)", flush=True)
    total = time.time() - start

    import analyze
    import importlib
    importlib.reload(analyze)  # load the fresh seed files
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        analyze.main()
    text = buf.getvalue()
    print(text)
    with open("results/analysis_output.txt", "w") as f:
        f.write(text)

    import numpy
    with open("results/run_info.txt", "w") as f:
        f.write("Axolotl-Q 10-seed study: where it ran\n")
        f.write(f"finished: {datetime.datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"computer: {platform.node()}\n")
        f.write(f"system: {platform.platform()}\n")
        f.write(f"processor: {platform.processor() or platform.machine()}\n")
        f.write(f"CPU cores used: {workers} of {os.cpu_count()}\n")
        f.write(f"python: {platform.python_version()}  numpy: {numpy.__version__}\n")
        f.write(f"seeds: {SEEDS}\n")
        f.write(f"total time: {total/60:.1f} minutes\n")
    print(open("results/run_info.txt").read())
    print("ALL DONE. Send Claude the 'results' folder (or zip it).")


if __name__ == "__main__":
    mp.freeze_support()
    main()
