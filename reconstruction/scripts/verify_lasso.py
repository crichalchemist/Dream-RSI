"""Score the paper's Appendix C Lasso solver with SimpleTES's own evaluator.

Dream-RSI's Lasso task follows SimpleTES (same 17 synthetic instances, same
CPP_CODE/COMPILE_FLAGS wire format), and SimpleTES publishes that evaluator.
This script runs it, unmodified, on three programs:

    dream_rsi      generated/lasso_path_dream_rsi.py  (paper Listing 3)
    glmnet_port    SimpleTES seed program (C++ port of glmnet)
    simpletes_best SimpleTES's released best program

SimpleTES is AGPL-3.0, so it is not vendored: clone it and pass its path.

    git clone --depth 1 https://github.com/wq-will/SimpleTES
    python scripts/verify_lasso.py --simpletes ./SimpleTES [--repeats 2]

The evaluator is copied into a temp dir before import because the Eigen tree
vendored in SimpleTES lacks Eigen/Core (its .gitignore drops it); from the
temp dir the evaluator falls back to -I/usr/include/eigen3 (libeigen3-dev).

--downstream times held-out datasets of the paper's Fig. 3(a) with
SimpleTES's own benchmark() routine: DNA and Leukemia from the copies shipped
in SimpleTES, Colon and Duke fetched from the LIBSVM site, Gisette (22 MB
download, 85% split as in SimpleTES) with --gisette. RCV1 is left out: it is
several GB once densified.
"""

import argparse
import bz2
import json
import os
import shutil
import sys
import tempfile
import urllib.request

import numpy as np

from see.loader import load_module_from_path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LIBSVM = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/"
CACHE = os.path.join(os.path.expanduser("~"), ".cache", "dream-rsi-recon")
# Paper Fig. 3(a), ms on the authors' hardware:
# sklearn, SimpleTES, SimpleTES (dagger row, undefined in the paper), Dream-RSI Pro, Flash.
PAPER_MS = {
    "Gisette": (11275.2, 3141.9, 8651.0, 2841.0, 1091.9),
    "DNA": (93.8, 15.9, 37.6, 49.9, 31.4),
    "Leukemia": (227.2, 15.5, 28.2, 30.2, 21.0),
    "Colon": (229.8, 11.6, 19.5, 16.4, 12.2),
    "Duke Breast": (374.0, 18.1, 31.1, 32.5, 23.6),
}


def load_evaluator(simpletes_dir, workdir):
    src = os.path.join(simpletes_dir, "datasets", "numerical_tasks", "lasso_path", "evaluator.py")
    dst = os.path.join(workdir, "lasso_evaluator.py")
    shutil.copy(src, dst)
    # Its default memory cap is RAM/256 per child, too small to import sklearn.
    os.environ.setdefault("EVALUATOR_CONCURRENT_PROCESSES", "1")
    return load_module_from_path("lasso_evaluator", dst)


def _libsvm(fname):
    from sklearn.datasets import load_svmlight_file

    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, fname)
    if not os.path.exists(path):
        urllib.request.urlretrieve(LIBSVM + fname, path)
    X, y = load_svmlight_file(bz2.open(path))
    return np.asarray(X.todense(), dtype=np.float64), y.astype(np.float64)


def downstream(simpletes_dir, programs, n_reps, gisette=False):
    task = os.path.join(simpletes_dir, "datasets", "numerical_tasks", "lasso_path")
    # also pins OMP/BLAS threads to 1, as SimpleTES does
    gr = load_module_from_path(
        "simpletes_generate_results", os.path.join(task, "generate_results.py")
    )

    def npz(name):
        return tuple(np.load(os.path.join(task, "eval_data", name))[k] for k in "Xy")

    datasets = {
        "DNA": lambda: npz("real_dna.npz"),
        "Leukemia": lambda: npz("real_leukemia.npz"),
        "Colon": lambda: _libsvm("colon-cancer.bz2"),
        "Duke Breast": lambda: _libsvm("duke.bz2"),
    }
    if gisette:
        from sklearn.model_selection import train_test_split

        def _gisette():
            X, y = _libsvm("gisette_scale.bz2")
            X, _, y, _ = train_test_split(X, y, test_size=0.15, random_state=42)
            return X, y

        datasets = {"Gisette": _gisette, **datasets}
    simpletes_fn, err1 = gr.load_solver(programs["simpletes_best"])
    dream_fn, err2 = gr.load_solver(programs["dream_rsi"])
    if err1 or err2:
        sys.exit(f"solver load failed: {err1 or err2}")
    print(
        f"\n{'dataset':<14}{'shape':>13}{'sklearn':>9}{'SimpleTES':>11}{'Dream-RSI':>11}"
        f"{'gap(D)':>9}  S/D here | paper S/D: Pro Flash  S+/D: Pro Flash  sk/D here Pro Flash"
    )
    rows = {}
    for name, load in datasets.items():
        X, y = load()
        sk_ms, s, d = gr.benchmark(X, y, simpletes_fn, dream_fn, n_reps, 1e-6)
        paper = PAPER_MS[name]
        ok = s["valid"] and d["valid"]
        ratio = s["ms"] / d["ms"] if ok else float("nan")
        sk, st, std, pro, fl = paper
        print(
            f"{name:<14}{X.shape!s:>13}{sk_ms:>9.1f}{s['ms']:>11.1f}{d['ms']:>11.1f}"
            f"{d['gap']:>9.1e}{ratio:>10.2f} |{st / pro:>15.2f}{st / fl:>6.2f}{std / pro:>10.2f}"
            f"{std / fl:>6.2f}{sk_ms / d['ms']:>11.2f}{sk / pro:>5.1f}{sk / fl:>6.1f}"
        )
        rows[name] = {"shape": X.shape, "sklearn_ms": sk_ms, "simpletes": s, "dream_rsi": d}
    print(
        "S/D = SimpleTES ms / Dream-RSI ms (above 1: the paper's solver is faster); "
        "S+ = the paper's dagger row; sk/D = sklearn ms / Dream-RSI ms"
    )
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=(__doc__ or "").partition("\n")[0])
    ap.add_argument("--simpletes", required=True, help="path to a SimpleTES checkout")
    ap.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="evaluator runs per program (each draws fresh instances)",
    )
    ap.add_argument(
        "--downstream",
        action="store_true",
        help="time the held-out datasets instead of the search instances",
    )
    ap.add_argument("--gisette", action="store_true", help="with --downstream, add Gisette")
    ap.add_argument("--json", help="write the per-problem results here")
    args = ap.parse_args(argv)

    dream = os.path.join(ROOT, "generated", "lasso_path_dream_rsi.py")
    if not os.path.exists(dream):
        sys.exit("run tools/extract_listings.py first")
    task = os.path.join(args.simpletes, "datasets", "numerical_tasks", "lasso_path")
    programs = {
        "dream_rsi": dream,
        "glmnet_port": os.path.join(task, "init_program.py"),
        "simpletes_best": os.path.join(
            args.simpletes,
            "best_results",
            "scientific_algorithms",
            "lasso_regularization_path",
            "lasso_regularization_path_best.py",
        ),
    }
    if args.downstream:
        rows = downstream(args.simpletes, programs, n_reps=5, gisette=args.gisette)
        if args.json:
            with open(args.json, "w") as f:
                json.dump(rows, f, indent=1, default=float)
        return
    with tempfile.TemporaryDirectory() as workdir:
        ev = load_evaluator(args.simpletes, workdir)
        labels = [f"n{n}_p{p}_{g}" for (n, p, g) in ev.PROBLEM_SIZES]
        results = {}
        for name, path in programs.items():
            runs = []
            for _ in range(args.repeats):
                r = ev.evaluate(path)
                runs.append(r)
            results[name] = runs

    print(f"\n{'program':<16}{'run':>4}{'valid':>8}{'geo-mean ms':>13}{'score':>10}{'max gap':>11}")
    for name, runs in results.items():
        for i, r in enumerate(runs):
            gaps = [p["max_gap"] for p in r.get("problems", [])]
            print(
                f"{name:<16}{i:>4}{r['n_valid']:>5}/{r['n_total']:<2}"
                f"{r['geo_mean_sol_ms']:>13.2f}{r['combined_score']:>10.4f}"
                f"{(max(gaps) if gaps else float('nan')):>11.1e}"
                + (f"  ERROR {r['error'][:60]}" if r.get("error") else "")
            )
    base = results["glmnet_port"]
    print("\nspeed-up over the glmnet port (ratio of geo-mean ms, per run):")
    for name, runs in results.items():
        ratios = [
            b["geo_mean_sol_ms"] / r["geo_mean_sol_ms"] for r, b in zip(runs, base, strict=True)
        ]
        print(f"  {name:<16}" + "  ".join(f"{x:.2f}x" for x in ratios))
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"labels": labels, "results": results}, f, indent=1, default=float)


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    main()
