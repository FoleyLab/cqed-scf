"""
Dense vs density-fitted QED-SAPT0: wall time, CPU time and peak memory across
a basis-set series for the water--methylamine dimer.

Measurement design
------------------
* Every (basis, backend, repeat) job runs in a fresh Python subprocess.
  Peak resident memory (ru_maxrss) is a per-process high-water mark, and
  Psi4/NumPy allocations are not visible to Python-level tools such as
  tracemalloc, so a fresh process is the only way to get an honest peak for
  each job. Running in a child process also lets the driver survive an
  out-of-memory kill.
* Inside the child, only the sapt0_components() call is timed. Interpreter
  start-up, imports, and geometry/config setup are excluded. The RSS just
  before the call is recorded as a baseline, so you can report either
  absolute peak RSS or peak minus baseline (the memory the calculation
  itself added).
* A background thread samples RSS every --sample-interval seconds as a
  cross-check on ru_maxrss and to give a memory-vs-time trace.
* The dense backend is skipped once its predicted peak exceeds
  --max-dense-gb, or after it fails, is OOM-killed, or times out. The
  prediction is calibrated on-the-fly from completed dense runs
  (peak scales as N_bf^4). DF continues through the whole series.

Usage
-----
    python benchmark_dense_vs_df.py                      # run the series
    python benchmark_dense_vs_df.py --threads 8 --repeats 3
    python benchmark_dense_vs_df.py --resume             # continue after interruption
    python benchmark_dense_vs_df.py --summary-only       # tables from existing results
    python benchmark_dense_vs_df.py --plot               # tables + figure

Results are appended one JSON record per line to --results (default
benchmark_results.jsonl), so an interrupted run loses at most the current job.
"""

import argparse
import json
import os
import platform
import resource
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime

EH_TO_KCAL = 627.5094740631
COMPONENT_KEYS = ("elst10", "exch10", "ind20", "exch_ind20", "disp20", "exch_disp20", "total")

DIMER = """
0 1
O   -0.687464896  -0.111744327  -0.019625472
H   -1.046121544   0.775938208   0.012706845
H    0.274042519   0.025850654  -0.003497262
--
0 1
N    2.787113199   0.125007400   0.008492726
H    3.082477630  -0.427630575  -0.786298137
H    3.097193694  -0.385713691   0.825352219
C    3.446448476   1.433371365  -0.031748912
H    3.135906054   2.015096325   0.832766508
H    4.537757766   1.394076393  -0.040704580
H    3.119736204   1.969288834  -0.919572724
symmetry c1
no_com
no_reorient
"""

DEFAULT_BASES = [
    "cc-pVDZ", "jun-cc-pVDZ", "aug-cc-pVDZ" ] #"cc-pVTZ", "jun-cc-pVTZ", "aug-cc-pVTZ",
#]


# =============================================================================
# Memory helpers (used in the worker)
# =============================================================================

def current_rss_bytes():
    """Current resident set size of this process, in bytes."""
    try:  # Linux: cheapest and dependency-free
        with open("/proc/self/statm") as fh:
            return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except OSError:
        import psutil  # macOS / other
        return psutil.Process().memory_info().rss


def peak_rss_bytes():
    """Kernel-tracked high-water mark of RSS for this process, in bytes."""
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return maxrss if sys.platform == "darwin" else maxrss * 1024  # Linux reports KiB


def cpu_seconds():
    """User + system CPU time of this process, summed over all threads."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_utime + ru.ru_stime


class RSSSampler(threading.Thread):
    """Samples RSS periodically; gives a trace and a sampled peak."""

    def __init__(self, interval):
        super().__init__(daemon=True)
        self.interval = interval
        self.trace = []  # (seconds since start, RSS MB)
        self._halt = threading.Event()
        self._t0 = None

    def run(self):
        self._t0 = time.perf_counter()
        while not self._halt.is_set():
            self.trace.append((time.perf_counter() - self._t0, current_rss_bytes() / 2**20))
            self._halt.wait(self.interval)

    def stop(self):
        self._halt.set()
        self.join()
        return max((m for _, m in self.trace), default=0.0)


# =============================================================================
# Worker: one job in a fresh process
# =============================================================================

def run_worker(job, result_path):
    import gc

    import numpy as np
    import psi4
    from cqed_scf import CQEDCalculator, CQEDConfig

    psi4.set_num_threads(job["threads"])
    psi4.set_memory(f"{job['psi4_mem_gb']} GB")
    psi4.core.set_output_file(job["psi4_out"], False)

    dense = job["backend"] == "full_eri"
    psi4_options = {
        "basis": job["basis"],
        "scf_type": job["dense_scf_type"] if dense else "df",
        "e_convergence": 1e-10,
        "d_convergence": 1e-8,
    }
    config = CQEDConfig(
        lambda_vector=np.array(job["lambda_vector"]),
        omega=0.1,  # QED-SAPT0 is omega-independent; kept for config completeness
        psi4_options=psi4_options,
        reference="rhf",
        functional=None,
        density_fitting=not dense,
        charge=0,
        multiplicity=1,
        dispersion_policy="none",
        debug=False,  # printing would pollute timings
        quiet=True,
    )
    calc = CQEDCalculator(config=config)
    mol = psi4.geometry(DIMER)

    kwargs = dict(integral_backend=job["backend"], include_cavity_terms=True)
    if job["frame"]:
        kwargs["monomer_reference_frame"] = job["frame"]

    gc.collect()
    baseline = current_rss_bytes()
    sampler = RSSSampler(job["sample_interval"])
    sampler.start()
    c0, t0 = cpu_seconds(), time.perf_counter()

    comps = calc.sapt0_components(mol, **kwargs)

    wall = time.perf_counter() - t0
    cpu = cpu_seconds() - c0
    sampled_peak = sampler.stop()
    peak = peak_rss_bytes()

    result = {
        "status": "ok",
        "wall_s": wall,
        "cpu_s": cpu,
        "baseline_rss_mb": baseline / 2**20,
        "peak_rss_mb": peak / 2**20,
        "peak_minus_baseline_mb": (peak - baseline) / 2**20,
        "sampled_peak_rss_mb": sampled_peak,
        "rss_trace": sampler.trace[:: max(1, len(sampler.trace) // 500)],  # cap size
        "components": {k: float(getattr(comps, k)) for k in COMPONENT_KEYS},
        "psi4_version": psi4.__version__,
    }
    try:
        import cqed_scf
        result["cqed_scf_version"] = getattr(cqed_scf, "__version__", "unknown")
    except Exception:
        pass
    with open(result_path, "w") as fh:
        json.dump(result, fh)
    psi4.core.clean()


# =============================================================================
# Driver
# =============================================================================

def physical_memory_gb():
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30


def basis_sizes(bases):
    """N_bf of the dimer for each basis (computed once, in the driver)."""
    import psi4
    psi4.core.be_quiet()
    mol = psi4.geometry(DIMER)
    sizes = {}
    for b in bases:
        sizes[b] = psi4.core.BasisSet.build(mol, "ORBITAL", b, quiet=True).nbf()
    psi4.core.clean()
    return sizes


def predict_dense_peak_gb(nbf, completed, n_tensors):
    """Predicted absolute peak RSS (GB) of a dense job.

    Uses the larger of a static model (n_tensors * N^4 doubles) and an N^4
    extrapolation from the largest completed dense run on this machine.
    """
    static = n_tensors * 8 * nbf**4 / 2**30
    if not completed:
        return static + 0.5  # rough allowance for interpreter + Psi4 baseline
    ref = max(completed, key=lambda r: r["nbf"])
    extra = ref["peak_minus_baseline_mb"] / 1024 * (nbf / ref["nbf"]) ** 4
    return max(static, extra) + ref["baseline_rss_mb"] / 1024


def run_job(job, timeout_s):
    """Launch one worker subprocess and return its result record."""
    job["workdir"] = os.path.abspath(job["workdir"])
    os.makedirs(job["workdir"], exist_ok=True)
    result_path = os.path.join(job["workdir"], "result.json")
    if os.path.exists(result_path):
        os.remove(result_path)

    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[var] = str(job["threads"])

    cmd = [sys.executable, os.path.abspath(__file__), "--worker",
           json.dumps(job), result_path]
    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, env=env, cwd=job["workdir"], timeout=timeout_s,
                              capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "elapsed_s": time.perf_counter() - start}

    if proc.returncode == 0 and os.path.exists(result_path):
        with open(result_path) as fh:
            return json.load(fh)
    if proc.returncode < 0:  # killed by a signal; -9 is the usual OOM-killer signature
        status = "killed" if proc.returncode == -9 else f"signal{-proc.returncode}"
    elif "MemoryError" in proc.stderr:
        status = "memory_error"
    else:
        status = "failed"
    return {"status": status, "returncode": proc.returncode,
            "stderr_tail": proc.stderr[-2000:]}


def load_results(path):
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def append_result(path, record):
    with open(path, "a") as fh:
        fh.write(json.dumps(record) + "\n")


def run_series(args):
    bases = args.bases
    print("Determining basis-set sizes ...", flush=True)
    sizes = basis_sizes(bases)
    bases = sorted(bases, key=sizes.get)

    meta = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "physical_memory_gb": round(physical_memory_gb(), 1),
        "threads": args.threads,
        "lambda_vector": args.lambda_vector,
        "frame": args.frame,
        "dense_scf_type": args.dense_scf_type,
        "started": datetime.now().isoformat(timespec="seconds"),
    }
    print(json.dumps(meta, indent=2))

    previous = load_results(args.results) if args.resume else []
    if not args.resume and os.path.exists(args.results):
        os.rename(args.results, args.results + ".bak")
        print(f"(moved existing {args.results} to {args.results}.bak)")
    done = {(r["basis"], r["backend"], r["repeat"]) for r in previous if r["status"] == "ok"}
    dense_completed = [r for r in previous if r["backend"] == "full_eri" and r["status"] == "ok"]
    dense_stopped = any(r["backend"] == "full_eri" and r["status"] not in ("ok",)
                        for r in previous)

    for basis in bases:
        nbf = sizes[basis]
        for backend in ("df", "full_eri"):
            for rep in range(args.repeats):
                key = (basis, backend, rep)
                if key in done:
                    continue
                record = {"basis": basis, "nbf": nbf, "backend": backend, "repeat": rep, **meta}

                if backend == "full_eri":
                    predicted = predict_dense_peak_gb(nbf, dense_completed, args.dense_tensors)
                    record["predicted_peak_gb"] = round(predicted, 2)
                    if dense_stopped:
                        record["status"] = "skipped_after_failure"
                        append_result(args.results, record)
                        print(f"{basis:>12} N={nbf:4d} {backend:>8}: skipped (dense series stopped earlier)")
                        break
                    if predicted > args.max_dense_gb:
                        record["status"] = "skipped_predicted_memory"
                        append_result(args.results, record)
                        print(f"{basis:>12} N={nbf:4d} {backend:>8}: skipped "
                              f"(predicted {predicted:.1f} GB > cap {args.max_dense_gb:.1f} GB)")
                        dense_stopped = True
                        break

                job = {
                    "basis": basis, "backend": backend, "threads": args.threads,
                    "psi4_mem_gb": args.psi4_mem_gb, "lambda_vector": args.lambda_vector,
                    "frame": args.frame, "dense_scf_type": args.dense_scf_type,
                    "sample_interval": args.sample_interval,
                    "workdir": os.path.join(args.workdir, f"{basis}_{backend}_r{rep}"),
                    "psi4_out": "psi4.out",
                }
                print(f"{basis:>12} N={nbf:4d} {backend:>8} rep {rep}: running ...", end="", flush=True)
                outcome = run_job(job, args.timeout_hours * 3600)
                record.update(outcome)
                append_result(args.results, record)

                if record["status"] == "ok":
                    print(f" {record['wall_s']:9.1f} s  peak {record['peak_rss_mb']:9.0f} MB "
                          f"(+{record['peak_minus_baseline_mb']:.0f})")
                    if backend == "full_eri":
                        dense_completed.append(record)
                else:
                    print(f" {record['status']}")
                    if backend == "full_eri":
                        dense_stopped = True
                        break
                    # A DF failure is unexpected: report and continue with the series.
                    if "stderr_tail" in record:
                        print(record["stderr_tail"][-800:])


# =============================================================================
# Reporting
# =============================================================================

def aggregate(records):
    """Collapse repeats: min wall time, median CPU, max peak memory."""
    import statistics
    groups = {}
    for r in records:
        if r.get("status") == "ok":
            groups.setdefault((r["basis"], r["backend"]), []).append(r)
    out = {}
    for key, rs in groups.items():
        out[key] = {
            "nbf": rs[0]["nbf"],
            "n": len(rs),
            "wall_s": min(r["wall_s"] for r in rs),
            "wall_spread_s": max(r["wall_s"] for r in rs) - min(r["wall_s"] for r in rs),
            "cpu_s": statistics.median(r["cpu_s"] for r in rs),
            "peak_rss_mb": max(r["peak_rss_mb"] for r in rs),
            "peak_minus_baseline_mb": max(r["peak_minus_baseline_mb"] for r in rs),
            "sampled_peak_rss_mb": max(r["sampled_peak_rss_mb"] for r in rs),
            "components": rs[0]["components"],
        }
    return out


def print_summary(path):
    records = load_results(path)
    if not records:
        print(f"No results in {path}")
        return
    agg = aggregate(records)
    bases = sorted({r["basis"] for r in records}, key=lambda b: next(r["nbf"] for r in records if r["basis"] == b))

    print("\nResource usage (wall = min over repeats; peak = max over repeats)")
    print(f"{'basis':>12} {'N_bf':>5} {'backend':>8} {'wall / s':>10} {'CPU / s':>10} "
          f"{'peak / MB':>10} {'+calc / MB':>10} {'speedup':>8} {'mem ratio':>9}")
    for b in bases:
        df, de = agg.get((b, "df")), agg.get((b, "full_eri"))
        for backend, a in (("df", df), ("full_eri", de)):
            if a is None:
                status = next((r["status"] for r in records
                               if r["basis"] == b and r["backend"] == backend), "not run")
                nbf = next(r["nbf"] for r in records if r["basis"] == b)
                print(f"{b:>12} {nbf:5d} {backend:>8} {status:>10}")
                continue
            extra = ""
            if backend == "full_eri" and df is not None:
                extra = (f" {a['wall_s'] / df['wall_s']:8.1f}"
                         f" {a['peak_minus_baseline_mb'] / max(df['peak_minus_baseline_mb'], 1):9.1f}")
            print(f"{b:>12} {a['nbf']:5d} {backend:>8} {a['wall_s']:10.1f} {a['cpu_s']:10.1f} "
                  f"{a['peak_rss_mb']:10.0f} {a['peak_minus_baseline_mb']:10.0f}{extra}")

    both = [b for b in bases if (b, "df") in agg and (b, "full_eri") in agg]
    if both:
        print("\nDF error, DF minus dense (kcal/mol)")
        print(f"{'basis':>12} " + " ".join(f"{k:>11}" for k in COMPONENT_KEYS))
        for b in both:
            df_c, de_c = agg[(b, "df")]["components"], agg[(b, "full_eri")]["components"]
            print(f"{b:>12} " + " ".join(
                f"{EH_TO_KCAL * (df_c[k] - de_c[k]):11.2e}" for k in COMPONENT_KEYS))

    csv_path = os.path.splitext(path)[0] + "_summary.csv"
    with open(csv_path, "w") as fh:
        fh.write("basis,nbf,backend,n_repeats,wall_s,wall_spread_s,cpu_s,peak_rss_mb,"
                 "peak_minus_baseline_mb,sampled_peak_rss_mb,"
                 + ",".join(COMPONENT_KEYS) + "\n")
        for (b, backend), a in sorted(agg.items(), key=lambda kv: (kv[1]["nbf"], kv[0][1])):
            fh.write(f"{b},{a['nbf']},{backend},{a['n']},{a['wall_s']:.3f},{a['wall_spread_s']:.3f},"
                     f"{a['cpu_s']:.3f},{a['peak_rss_mb']:.1f},{a['peak_minus_baseline_mb']:.1f},"
                     f"{a['sampled_peak_rss_mb']:.1f},"
                     + ",".join(f"{a['components'][k]:.12f}" for k in COMPONENT_KEYS) + "\n")
    print(f"\nwrote {csv_path}")


def plot(path, out="benchmark_dense_vs_df.pdf"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    records = load_results(path)
    agg = aggregate(records)
    fig, (ax_t, ax_m) = plt.subplots(1, 2, figsize=(9, 3.6))
    style = {"df": ("tab:blue", "o", "density-fitted"), "full_eri": ("tab:red", "s", "dense")}
    for backend, (color, marker, label) in style.items():
        pts = sorted((a["nbf"], a) for (b, be), a in agg.items() if be == backend)
        if not pts:
            continue
        n = [p[0] for p in pts]
        ax_t.loglog(n, [p[1]["wall_s"] for p in pts], marker=marker, color=color, label=label)
        ax_m.loglog(n, [p[1]["peak_minus_baseline_mb"] / 1024 for p in pts],
                    marker=marker, color=color, label=label)
    for r in records:  # mark dense jobs that were skipped or failed, at their predicted peak
        if r["backend"] == "full_eri" and r.get("status") != "ok" and "predicted_peak_gb" in r:
            ax_m.loglog(r["nbf"], r["predicted_peak_gb"], marker="x", color="tab:red", ls="none")
    ax_t.set_ylabel("wall time / s")
    ax_m.set_ylabel("peak memory added by calculation / GB")
    for ax in (ax_t, ax_m):
        ax.set_xlabel(r"$N_\mathrm{bf}$")
        ax.grid(alpha=0.3, which="both")
    ax_t.legend()
    fig.tight_layout()
    fig.savefig(out)
    print(f"wrote {out}")


# =============================================================================
# CLI
# =============================================================================

def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--worker":
        run_worker(json.loads(sys.argv[2]), sys.argv[3])
        return

    phys = physical_memory_gb()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bases", nargs="+", default=DEFAULT_BASES)
    ap.add_argument("--threads", type=int, default=1,
                    help="threads for Psi4/BLAS in every job (keep fixed across the series)")
    ap.add_argument("--repeats", type=int, default=1,
                    help="fresh-process repeats per job; min wall time is reported")
    ap.add_argument("--lambda-vector", type=float, nargs=3, default=[0.0, 0.0, 0.05])
    ap.add_argument("--frame", default="monomer_com",
                    help="monomer_reference_frame passed to sapt0_components ('' to omit)")
    ap.add_argument("--dense-scf-type", default="pk",
                    help="scf_type for the dense pipeline (pk = fully exact; df isolates the SAPT backend)")
    ap.add_argument("--max-dense-gb", type=float, default=0.8 * phys,
                    help=f"skip dense jobs predicted above this peak RSS (default 80%% of {phys:.0f} GB)")
    ap.add_argument("--dense-tensors", type=float, default=4.0,
                    help="N_bf^4 tensors assumed by the static dense memory model before calibration")
    ap.add_argument("--psi4-mem-gb", type=float, default=round(0.5 * phys, 1),
                    help="psi4.set_memory value (an algorithm hint; it does not cap NumPy arrays)")
    ap.add_argument("--timeout-hours", type=float, default=12.0)
    ap.add_argument("--sample-interval", type=float, default=0.05)
    ap.add_argument("--results", default="benchmark_results.jsonl")
    ap.add_argument("--workdir", default="benchmark_runs")
    ap.add_argument("--resume", action="store_true", help="skip jobs already completed in --results")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    args.frame = args.frame or None

    if not args.summary_only:
        run_series(args)
    print_summary(args.results)
    if args.plot:
        plot(args.results)


if __name__ == "__main__":
    main()
