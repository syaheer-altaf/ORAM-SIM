#!/usr/bin/env python3
"""
Analyze Experiment 3 for the fixed stability-boundary configuration

    Z(d) = 4,  d < L-2
           2,  d = L-2
           0,  d > L-2

The input histogram is expected to contain columns

    L,k,m,stash,count

where k may remain in the simulator output, but it is not used in the
analysis or shown in the resulting tables/figures.

Outputs:
  1. exp3_required_stash.csv
       Long-form empirical R_lambda values for every (L,m).

  2. exp3_required_stash_table.csv
       Pivot table with rows L and columns m.

  3. exp3_tail_security_L<...>.png
       Empirical stash-tail curves at one representative tree height.
       The vertical axis is log2 Pr[S > R]. Horizontal dotted lines mark
       security levels lambda = 10,...,17.

Usage:
    python analyze_exp3.py [results/exp3/exp3_hist.csv]
    python analyze_exp3.py exp3_hist.csv --lam 15
    python analyze_exp3.py exp3_hist.csv --lam 15 --tail-L 20
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MIN_TAIL = 10
TAIL_R_MAX = 12
TAIL_LAMBDA_MIN = 10
TAIL_LAMBDA_MAX = 17


def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "hist",
        nargs="?",
        type=Path,
        default=Path("./results/exp3/exp3_hist.csv"),
        help="stash histogram CSV",
    )

    ap.add_argument(
        "--lam",
        type=int,
        help=(
            "security parameter used to compute R_lambda; "
            "default is chosen from MIN_TAIL"
        ),
    )

    ap.add_argument(
        "--tail-L",
        type=int,
        help=(
            "tree height used in the tail figure; "
            "default is the largest available L"
        ),
    )

    ap.add_argument(
        "--out",
        type=Path,
        help="output directory; defaults to <hist-dir>/exp3_analysis",
    )

    return ap.parse_args()


def main():
    args = parse_args()

    hist = pd.read_csv(args.hist)

    required = {"L", "m", "stash", "count"}
    missing = required - set(hist.columns)

    if missing:
        raise ValueError(
            "histogram CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )

    # This analysis assumes one fixed bucket configuration.
    # The simulator may still write a k column, but it is not used.
    if "k" in hist.columns and hist["k"].nunique() > 1:
        raise ValueError(
            "input histogram contains more than one bucket configuration; "
            "provide the histogram for the fixed stability-boundary run only"
        )

    # tails[(L,m)] =
    #     (number of samples, Pr[S > R] for R = 0,1,...)
    tails = {}

    for key, g in hist.groupby(["L", "m"]):
        max_stash = int(g["stash"].max())

        counts = np.zeros(
            max_stash + 1,
            dtype=np.int64,
        )

        counts[g["stash"].astype(int)] = g["count"].astype(np.int64)

        n = int(counts.sum())

        # tail[R] = Pr[S > R]
        tail = (n - np.cumsum(counts)) / n

        tails[key] = (n, tail)

    # Conservative automatically resolvable lambda.
    n_min = min(n for n, _ in tails.values())

    lam_max_resolved = int(
        np.floor(np.log2(n_min / MIN_TAIL))
    )

    lam = (
        args.lam
        if args.lam is not None
        else lam_max_resolved
    )

    print(
        f"fewest samples n = {n_min} "
        f"-> lambda <= {lam_max_resolved} "
        f"for MIN_TAIL={MIN_TAIL}; "
        f"using lambda={lam}"
    )

    if lam > lam_max_resolved:
        print(
            f"warning: lambda={lam} is above the conservative "
            f"MIN_TAIL={MIN_TAIL} resolution criterion"
        )

    # --------------------------------------------------------------
    # Compute empirical R_lambda for every (L,m).
    #
    # R_lambda = min { R : Pr[S > R] <= 2^{-lambda} }.
    # --------------------------------------------------------------

    rows = []

    threshold = 2.0 ** (-lam)

    for (L, m), (n, p) in tails.items():

        hits = np.flatnonzero(
            p <= threshold
        )

        R = (
            int(hits[0])
            if len(hits)
            else np.nan
        )

        rows.append(
            {
                "L": int(L),
                "m": int(m),
                "samples": n,
                "R_min": R,
                "max_stash": len(p) - 1,
            }
        )

    res = (
        pd.DataFrame(rows)
        .sort_values(["L", "m"])
    )

    # Output directory.
    out = (
        args.out
        if args.out is not None
        else args.hist.parent / "exp3_analysis"
    )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Long-form results.
    res.to_csv(
        out / "exp3_required_stash.csv",
        index=False,
    )

    Ls = sorted(
        res["L"].unique()
    )

    ms = sorted(
        res["m"].unique()
    )

    # --------------------------------------------------------------
    # Compact R_lambda table.
    #
    # Rows:    L
    # Columns: m
    # --------------------------------------------------------------

    table = (
        res.pivot(
            index="L",
            columns="m",
            values="R_min",
        )
        .reindex(
            index=Ls,
            columns=ms,
        )
    )

    table.to_csv(
        out / "exp3_required_stash_table.csv"
    )

    print(
        f"\nEmpirical R_lambda table for lambda={lam}:"
    )

    print(
        table.to_string()
    )

    # --------------------------------------------------------------
    # Tail figure.
    # --------------------------------------------------------------

    L_tail = (
        args.tail_L
        if args.tail_L is not None
        else max(Ls)
    )

    if L_tail not in Ls:
        raise ValueError(
            f"tail figure requested L={L_tail}, "
            "but that tree height is not present "
            "in the histogram"
        )

    colors = dict(
        zip(
            ms,
            plt.cm.Blues(
                np.linspace(
                    0.35,
                    0.85,
                    len(ms),
                )
            ),
        )
    )

    fig, ax = plt.subplots(
        figsize=(7.2, 4.8)
    )

    for m in ms:

        key = (L_tail, m)

        if key not in tails:
            continue

        _, p = tails[key]

        # Plot only R = 0,...,TAIL_R_MAX.
        R = np.arange(
            min(
                len(p),
                TAIL_R_MAX + 1,
            )
        )

        # log2(0) is undefined, so omit zero-tail points.
        positive = p[R] > 0

        R = R[positive]

        log_tail = np.log2(
            p[R]
        )

        ax.plot(
            R,
            log_tail,
            marker="o",
            markersize=3.5,
            linewidth=1.5,
            color=colors[m],
            label=fr"$m={m}$",
        )

    ax.set_xlim(
        0,
        TAIL_R_MAX,
    )

    ax.set_xticks(
        np.arange(
            0,
            TAIL_R_MAX + 1,
            1,
        )
    )

    # A target security level lambda satisfies
    #
    #     Pr[S > R] <= 2^{-lambda}
    #
    # which is equivalent to
    #
    #     log2 Pr[S > R] <= -lambda.
    #
    # Mark those thresholds directly on the
    # log2-probability scale.

    y_top = -5

    y_bottom = (
        -TAIL_LAMBDA_MAX - 0.5
    )

    ax.set_ylim(
        y_bottom,
        y_top,
    )

    ax.set_yticks(
        np.arange(
            -TAIL_LAMBDA_MAX,
            y_top + 1,
            2,
        )
    )

    for lam_line in range(
        TAIL_LAMBDA_MIN,
        TAIL_LAMBDA_MAX + 1,
    ):

        y = -lam_line

        ax.axhline(
            y,
            linestyle=":",
            linewidth=0.9,
            alpha=0.45,
        )

        ax.text(
            TAIL_R_MAX - 0.12,
            y + 0.08,
            fr"$\lambda={lam_line}$",
            ha="right",
            va="bottom",
            fontsize=7,
            alpha=0.8,
        )

    ax.set_xlabel(
        r"stash bound $R$"
    )

    ax.set_ylabel(
        r"$\log_2 \Pr[S>R]$"
    )

    ax.set_title(
        fr"Empirical stash tails ($L={L_tail}$)"
    )

    ax.grid(
        axis="x",
        alpha=0.20,
    )

    ax.legend(
        ncol=2,
        fontsize=8,
    )

    fig.tight_layout()

    fig.savefig(
        out / f"exp3_tail_security_L{L_tail}.png",
        dpi=220,
    )

    plt.close(fig)

    print(
        "\nsaved:\n"
        f"  {out / 'exp3_required_stash.csv'}\n"
        f"  {out / 'exp3_required_stash_table.csv'}\n"
        f"  {out / f'exp3_tail_security_L{L_tail}.png'}"
    )


if __name__ == "__main__":
    main()