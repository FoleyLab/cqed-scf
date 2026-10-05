"""
Plot QED-SAPT0 benzene dimer slip-scan results written by
qed_sapt0_df_benzene_slip_scan.py (slip_scan_results.json).

Default: total interaction energy vs slip, one curve per cavity polarization.
--components: one panel per SAPT component (elst, exch, ind, disp, total).

Usage:
    python plot_benzene_slip_scan.py [--components] [--results FILE] [--out FILE] [--show]
"""

import argparse
import json

import matplotlib

EH_TO_KCAL = 627.5094740631

# (panel title, keys summed into it); "ind" and "disp" include their exchange parts.
COMPONENTS = [
    ("Electrostatics", ("elst10",)),
    ("Exchange", ("exch10",)),
    ("Induction", ("ind20", "exch_ind20")),
    ("Dispersion", ("disp20", "exch_disp20")),
    ("Total", ("total",)),
]

STYLE = {  # polarization label -> (color, linestyle)
    "none (lambda=0)": ("0.4", "--"),
    "x (slip)": ("tab:blue", "-"),
    "y (perp)": ("tab:green", "-"),
    "z (stack)": ("tab:red", "-"),
}


def load(path):
    with open(path) as fh:
        data = json.load(fh)
    # {label: {slip(float): {component: Eh}}}
    return {
        label: {float(slip): comps for slip, comps in series.items()}
        for label, series in data["results"].items()
    }, data.get("meta", {})


def curve(series, keys):
    slips = sorted(series)
    return slips, [EH_TO_KCAL * sum(series[s][k] for k in keys) for s in slips]


def plot(results_path="slip_scan_results.json", components=False, out=None, show=False):
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results, meta = load(results_path)
    subtitle = ", ".join(f"{k} = {v}" for k, v in meta.items())

    panels = COMPONENTS if components else [COMPONENTS[-1]]
    ncols = 3 if components else 1
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.0 * ncols, 3.8 * nrows), squeeze=False
    )

    for ax, (title, keys) in zip(axes.flat, panels):
        for label, series in results.items():
            color, ls = STYLE.get(label, (None, "-"))
            x, y = curve(series, keys)
            ax.plot(x, y, marker="o", ms=4, color=color, ls=ls, label=label)
        ax.set_title(title if components else "Total QED-SAPT0 interaction energy")
        ax.set_ylabel("E / kcal mol$^{-1}$")
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    for ax in axes.flat:
        if ax.get_visible():
            ax.set_xlabel("slip / Å")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    if components:
        fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.97, 0.12))
    else:
        axes.flat[0].legend()
    fig.suptitle(f"Benzene dimer slip scan  ({subtitle})" if subtitle else "Benzene dimer slip scan")
    fig.tight_layout()

    out = out or ("slip_scan_components.png" if components else "slip_scan_total.png")
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")
    if show:
        plt.show()
    plt.close(fig)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--components", action="store_true", help="plot each SAPT component")
    ap.add_argument("--results", default="slip_scan_results.json")
    ap.add_argument("--out", default=None)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    plot(args.results, args.components, args.out, args.show)
