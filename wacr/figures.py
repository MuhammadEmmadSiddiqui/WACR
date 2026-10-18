"""Paper figures. Drawing copied from the manuscript figure scripts; paths and imports only."""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, MultipleLocator, NullFormatter, ScalarFormatter
from matplotlib.transforms import blended_transform_factory

MM = 1.0 / 25.4
FULL_W = 190 * MM
ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
HEAD = 0.10
PAIRS = ("TinyFace", "XQLFW")

OKABE = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
}

SPLIT = "Split CP"
QTC = "QTC"
COND = "Conditional CP"
RAFR = "RA-FR"
WRCP_A = "WRCP"
DSCP = "DS-CP"
WACR_G1 = "WACR (Gamma = 1)"
WACR_G15 = "WACR (Gamma = 1.5)"
PAC_GREEN = OKABE["green"]
BLUE = OKABE["blue"]

WACR_PRESET_STYLE = {
    WACR_G1: dict(color="#7EB6D3", lw=1.15, ls="-", mk="o", ms=3.6, z=4),
    WACR_G15: dict(color="#3A8FBE", lw=1.70, ls="-", mk="s", ms=3.8, z=5),
}
SERIES_STYLE = {
    SPLIT: dict(color=OKABE["black"], lw=1.35, ls=(0, (1.15, 1.15)), mk="o", ms=3.4, z=2),
    DSCP: dict(color="#8C564B", lw=1.35, ls=(0, (5.0, 1.5)), mk="p", ms=3.8, z=3),
    WRCP_A: dict(color=OKABE["vermillion"], lw=1.40, ls=(0, (2.2, 1.0, 2.2, 1.0)), mk="^", ms=3.8, z=3),
    COND: dict(color=OKABE["orange"], lw=1.35, ls=(0, (1.0, 1.2)), mk="D", ms=3.4, z=3),
    QTC: dict(color=OKABE["purple"], lw=1.40, ls=(0, (4.0, 1.0, 1.0, 1.0, 1.0, 1.0)), mk="*", ms=4.4, z=3),
    RAFR: dict(color="0.35", lw=1.40, ls=(0, (2.4, 1.4)), mk="P", ms=4.0, z=3),
    "KMM": dict(color="#AA4499", lw=1.35, ls=(0, (4.0, 1.8)), mk="s", ms=3.6, z=3),
    "KLIEP": dict(color="#999933", lw=1.40, ls=(0, (3.0, 1.1, 1.0, 1.1)), mk="h", ms=4.0, z=3),
    "WCP-DC": dict(color="#882255", lw=1.35, ls=(0, (1.0, 1.5, 4.0, 1.5)), mk="v", ms=3.6, z=3),
    "PAC-CS": dict(color="#8C510A", lw=1.40, ls="-", mk="<", ms=3.8, z=4),
    "LTT-IW": dict(color="#4B2A6A", lw=1.40, ls="-", mk="d", ms=3.8, z=4),
    "WACR-PAC": dict(color=PAC_GREEN, lw=1.55, ls="-", mk="D", ms=4.0, z=5),
}
SERIES_STYLE.update(WACR_PRESET_STYLE)

A_MAIN_NONE = (SPLIT, QTC, COND, RAFR)
A_MAIN_WGT = ("PAC-CS", WRCP_A, "LTT-IW", DSCP)
A_MAIN_PROP = (WACR_G1, WACR_G15, "WACR-PAC")
A_MAIN_ALL = A_MAIN_NONE + A_MAIN_WGT + A_MAIN_PROP
A_MAIN_PROP_LW = {WACR_G1: 1.70, WACR_G15: 1.90, "WACR-PAC": 2.10}
A_MAIN_DISPLAY = {
    SPLIT: "Split CP",
    QTC: "QTC",
    COND: "Conditional CP",
    RAFR: "RA-FR",
    "PAC-CS": "PAC-CS",
    WRCP_A: "WRCP",
    "LTT-IW": "LTT-IW",
    DSCP: "DS-CP",
    WACR_G1: r"WACR ($\Gamma = 1$)",
    WACR_G15: r"WACR ($\Gamma = 1.5$)",
    "WACR-PAC": r"WACR-PAC ($\Gamma = 1$)",
}

B_MAIN_ORDER = (
    "WACR_PAC",
    "WACR_15",
    "WACR_1",
    "HDR_PROP",
    "DSCP",
    "LTT_PUB",
    "WRCP",
    "PACCS_PUB",
    "WCP_DC",
    "WCP_KLIEP",
    "WCP_KMM",
    "HDR_W",
    "RAFR",
    "COND",
    "QTC",
    "SPLIT",
    "HDR_NONE",
)
B_HEADERS = frozenset({"HDR_NONE", "HDR_W", "HDR_PROP"})
B_PROPOSED = frozenset({"WACR_1", "WACR_15", "WACR_PAC"})
B_MAIN_DISPLAY = {
    "HDR_NONE": "No weight correction",
    "HDR_W": "Weight correction",
    "HDR_PROP": "Proposed",
    "SPLIT": "Split CP",
    "QTC": "QTC",
    "COND": "Conditional CP",
    "RAFR": "RA-FR",
    "WCP_KMM": "WCP-KMM",
    "WCP_KLIEP": "WCP-KLIEP",
    "WCP_DC": "WCP-DC",
    "PACCS_PUB": "PAC-CS",
    "WRCP": "WRCP",
    "LTT_PUB": "LTT-IW",
    "DSCP": "DS-CP",
    "WACR_1": r"WACR ($\Gamma = 1$)",
    "WACR_15": r"WACR ($\Gamma = 1.5$)",
    "WACR_PAC": r"WACR-PAC ($\Gamma = 1$)",
}
B_MID_STYLE_NAME = {
    "SPLIT": SPLIT,
    "DSCP": DSCP,
    "WRCP": WRCP_A,
    "COND": COND,
    "QTC": QTC,
    "RAFR": RAFR,
    "PACCS_PUB": "PAC-CS",
    "LTT_PUB": "LTT-IW",
    "WACR_1": WACR_G1,
    "WACR_15": WACR_G15,
    "WACR_PAC": "WACR-PAC",
    "WCP_KMM": "KMM",
    "WCP_KLIEP": "KLIEP",
    "WCP_DC": "WCP-DC",
}
T1_TO_A_MAIN = {
    ("Split CP", "textbook mass 1 at +inf"): SPLIT,
    ("QTC", "min rule, Method B, conf=t"): QTC,
    ("Conditional CP", "one-hot 20 bins on t"): COND,
    ("RA-FR", "percentile budget"): RAFR,
    ("PAC-CS", "PS-W AdaFace"): "PAC-CS",
    ("WRCP", "XGB DINOv2 per-query, rho*"): WRCP_A,
    ("LTT-IW", "KMM QP WSR"): "LTT-IW",
    ("DS-CP", "XGB DINOv2 gamma=1"): DSCP,
    ("WACR", "Gamma=1"): WACR_G1,
    ("WACR", "Gamma=1.5"): WACR_G15,
    ("WACR-PAC", "Gamma=1, Bernstein"): "WACR-PAC",
}
T1_TO_B_MID = {
    "Split CP": "SPLIT",
    "QTC": "QTC",
    "Conditional CP": "COND",
    "RA-FR": "RAFR",
    "WCP-KMM": "WCP_KMM",
    "WCP-KLIEP": "WCP_KLIEP",
    "WCP-DC": "WCP_DC",
    "PAC-CS": "PACCS_PUB",
    "WRCP": "WRCP",
    "LTT-IW": "LTT_PUB",
    "DS-CP": "DSCP",
}


class _Hdr:
    def __init__(self, label: str) -> None:
        self._label = label

    def get_label(self) -> str:
        return self._label


class _NoHandle(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        line = Line2D([], [], linestyle="none", marker="none", color="none")
        line.set_transform(trans)
        return [line]


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.4,
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
        }
    )


def _f(x, default=float("nan")) -> float:
    if x in ("", None):
        return default
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _a_main_style(method: str) -> dict:
    sty = dict(SERIES_STYLE[method])
    if method in A_MAIN_PROP:
        sty["ls"] = "-"
        sty["lw"] = A_MAIN_PROP_LW[method]
    else:
        if sty["ls"] == "-":
            sty["ls"] = (0, (3.2, 1.4)) if method.startswith("PAC-CS") else (0, (1.15, 1.15))
        sty["lw"] = min(float(sty["lw"]), 1.40)
    return sty


def _draw_a_main(ax, xs, ys, method: str) -> None:
    sty = _a_main_style(method)
    ax.plot(
        list(xs),
        list(ys),
        color=sty["color"],
        lw=sty["lw"],
        ls=sty["ls"],
        marker=sty["mk"],
        ms=sty["ms"],
        zorder=sty["z"],
        solid_capstyle="butt",
    )


def _series(rows, cal, method):
    pts = sorted(
        (_f(r["alpha"]), _f(r["r_hat"]), _f(r["set_size"]))
        for r in rows
        if r["cal"] == cal and r["method"] == method and np.isfinite(_f(r["r_hat"]))
    )
    if not pts:
        return None
    xs, rh, cs = zip(*pts)
    return list(xs), list(rh), list(cs)


def _wacr_g1_band(rows, cal: str):
    pts = []
    for r in rows:
        if r.get("cal") != cal:
            continue
        if r.get("method") != WACR_G1:
            continue
        a = _f(r["alpha"])
        lo = _f(r.get("r_lo", ""))
        hi = _f(r.get("r_hi", ""))
        if np.isfinite(lo) and np.isfinite(hi):
            pts.append((a, lo, hi))
    if not pts:
        return None
    pts.sort()
    xs, lo, hi = zip(*pts)
    return list(xs), list(lo), list(hi)


def _style_alpha_axis(ax, *, xlabel: bool) -> None:
    ax.set_xlim(0.03, 0.52)
    ax.set_xticks(ALPHAS)
    if xlabel:
        ax.set_xticklabels([f"{a:.2f}" for a in ALPHAS])
        ax.set_xlabel(r"$\alpha$")
    else:
        ax.set_xticklabels([])
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.4)


def _style_log_c(ax, ymin, ymax) -> None:
    ax.set_yscale("log")
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=6))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, which="major", alpha=0.25, linestyle="--", linewidth=0.4)
    ax.grid(True, which="minor", alpha=0.12, linestyle=":", linewidth=0.3)


PDF_META = {"Title": "", "Author": "", "Creator": "", "Producer": "", "Subject": ""}
SCORE_COLOR = {"A": OKABE["blue"], "B": OKABE["vermillion"]}
SCORE_MK = {"A": "o", "B": "s"}
SCORE_LS = {"A": "-", "B": (0, (4.0, 1.6))}
DS_COLOR = {"TinyFace": OKABE["blue"], "XQLFW": OKABE["orange"], "SCFace": OKABE["green"]}
DS_MK = {"TinyFace": "o", "XQLFW": "s", "SCFace": "D"}
WEIGHT_STAIRS = {
    "Unweighted": dict(color=OKABE["black"], ls=(0, (1.15, 1.15)), lw=1.35),
    "KMM": dict(color=OKABE["orange"], ls=(0, (4.0, 1.8)), lw=1.40),
    "KLIEP": dict(color=OKABE["green"], ls=(0, (3.0, 1.1, 1.0, 1.1)), lw=1.40),
    "Domain classifier": dict(color="#882255", ls=(0, (1.0, 1.5, 4.0, 1.5)), lw=1.40),
    "Logistic-linear (ablation)": dict(color="0.55", ls=(0, (3.2, 1.6)), lw=1.15),
    "Logistic-quadratic": dict(color=OKABE["purple"], ls=(0, (2.4, 1.3)), lw=1.40),
    "WACR (histogram)": dict(color=OKABE["blue"], ls="-", lw=1.70),
}
WEIGHT_EMBED = ("Unweighted", "KMM", "KLIEP", "Domain classifier")
WEIGHT_ON_T = ("Logistic-linear (ablation)", "Logistic-quadratic", "WACR (histogram)")
WEIGHT_LEGEND = {
    "Logistic-linear (ablation)": "Logistic-linear, ablation",
    "WACR (histogram)": "WACR weights (histogram)",
}
SENS_HEAD_A = 0.10
SENS_ALPHAS = (0.05, 0.10, 0.20)
SENS_N_GAL = 585
SENS_B_CAP = 30.0
SENS_G_LO, SENS_G_HI = 1.0, 5.0
SENS_N_CAL = {"TinyFace": 2292, "XQLFW": 2296}
SENS_PAIRS = (("TinyFace", "TinyFace"), ("XQLFW", "XQLFW"))
ROW2_COLOR = {0.20: "#B08CC8", 0.10: "#7A4EA0", 0.05: "#3C1F5C"}
ROW2_MK = {0.05: "o", 0.10: "s", 0.20: "^"}
ADAPT_N_BINS = 20
ADAPT_N_GAL = 585
ADAPT_N_PROBE = 975
ADAPT_Y_HI = 1500.0
COLOR_HIT = "#009E73"
COLOR_MISS = "#D55E00"
COLOR_MEDIAN = "#0072B2"
DIST_M = {1: 4.20, 2: 2.60, 3: 1.00}
DIST_ORDER = (1, 2, 3)
DIST_LABELS = ("4.20 m", "2.60 m", "1.00 m")
ADAPT_SPEC = {
    "a": {
        "key": "a",
        "cal": "TinyFace",
        "gamma": 1.0,
        "title": r"(a) TinyFace $\to$ SCFace, $\Gamma=1$",
        "title_d": r"(d) TinyFace $\to$ SCFace, $\Gamma=1$",
    },
    "b": {
        "key": "b",
        "cal": "XQLFW",
        "gamma": 1.0,
        "title": r"(b) XQLFW $\to$ SCFace, $\Gamma=1$",
        "title_d": r"(e) XQLFW $\to$ SCFace, $\Gamma=1$",
    },
    "c": {
        "key": "c",
        "cal": "XQLFW",
        "gamma": 1.4,
        "title": r"(c) XQLFW $\to$ SCFace, $\hat{\Gamma}^{*}=1.4$",
        "title_d": r"(f) XQLFW $\to$ SCFace, $\hat{\Gamma}^{*}=1.4$",
    },
}
PLOT_TOL = 1e-6


def gamma_vac(alpha: float, n: int, b: float = 30.0) -> float:
    return float(np.sqrt(alpha * n / ((1.0 - alpha) * b)))


def _arr_eq(name: str, got, exp) -> None:
    g = np.asarray(got, dtype=np.float64)
    e = np.asarray(exp, dtype=np.float64)
    if g.shape != e.shape:
        raise SystemExit(f"plot vs csv {name}: shape {g.shape} vs {e.shape}")
    if g.size == 0:
        return
    g_ok = np.isfinite(g)
    e_ok = np.isfinite(e)
    if not np.array_equal(g_ok, e_ok):
        raise SystemExit(f"plot vs csv {name}: finite mask mismatch")
    if not np.any(g_ok):
        return
    d = float(np.max(np.abs(g[g_ok] - e[e_ok])))
    if d > PLOT_TOL:
        raise SystemExit(f"plot vs csv {name}: max |d|={d} > {PLOT_TOL}")


def save_pdf_png(fig: plt.Figure, pdf_path: Path, png_path: Path, *, dpi: int = 400) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.04,
        metadata=PDF_META,
        dpi=dpi,
    )
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    print(f"Wrote {png_path}", flush=True)
    print(f"Wrote {pdf_path}", flush=True)


def save_png(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=400, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    print(f"Wrote {path}", flush=True)


def plot_a_main(rows: list[dict], out_path: Path) -> None:
    apply_style()
    sizes = [
        _f(r["set_size"])
        for r in rows
        if r["method"] in A_MAIN_ALL and _f(r["set_size"]) > 0
    ]
    ylim_c = (0.03, max(max(sizes) * 1.35, 800.0) if sizes else 800.0)
    fig, axes = plt.subplots(2, 2, figsize=(FULL_W, 5.15), layout="constrained", sharex="col")
    titles = (
        (r"(a) TinyFace $\to$ SCFace", r"(b) TinyFace $\to$ SCFace"),
        (r"(c) XQLFW $\to$ SCFace", r"(d) XQLFW $\to$ SCFace"),
    )
    for r_i, cal in enumerate(PAIRS):
        ax_r, ax_c = axes[r_i]
        for name in A_MAIN_ALL:
            ser = _series(rows, cal, name)
            if ser is None:
                continue
            xs, rh, cs = ser
            _draw_a_main(ax_r, xs, rh, name)
            _draw_a_main(ax_c, xs, cs, name)
        ax_r.set_ylim(-0.02, 1.02)
        ax_r.set_ylabel(r"$\hat{R}$")
        _style_alpha_axis(ax_r, xlabel=(r_i == 1))
        ax_r.set_axisbelow(True)
        band = _wacr_g1_band(rows, cal)
        if band is not None:
            bx, blo, bhi = band
            c_g1 = WACR_PRESET_STYLE[WACR_G1]["color"]
            ax_r.fill_between(bx, blo, bhi, color=c_g1, alpha=0.3, linewidth=0, zorder=2)
            ax_r.plot(bx, blo, color=c_g1, lw=0.7, ls="-", zorder=2.2, solid_capstyle="butt")
            ax_r.plot(bx, bhi, color=c_g1, lw=0.7, ls="-", zorder=2.2, solid_capstyle="butt")
        ax_r.plot(
            [0.0, 0.55],
            [0.0, 0.55],
            ls=(0, (2.4, 1.3)),
            color="0.15",
            lw=1.15,
            zorder=8,
            clip_on=True,
            solid_capstyle="butt",
        )
        _style_log_c(ax_c, ylim_c[0], ylim_c[1])
        _style_alpha_axis(ax_c, xlabel=(r_i == 1))
        ax_c.set_ylabel(r"mean $|C|$")
        ax_c.set_axisbelow(True)
        ax_c.axhline(585.0, color="0.4", ls=":", lw=0.8, zorder=2)
        for ax, lab in zip((ax_r, ax_c), titles[r_i]):
            ax.set_title(lab, loc="left", fontsize=8, pad=2.5)
    groups = (
        ("No weight correction", A_MAIN_NONE),
        ("Weight correction", A_MAIN_WGT),
        ("Proposed", A_MAIN_PROP),
    )
    packed_groups = []
    for title, names in groups:
        hs = []
        for name in names:
            sty = _a_main_style(name)
            hs.append(
                Line2D(
                    [0],
                    [0],
                    color=sty["color"],
                    lw=sty["lw"],
                    ls=sty["ls"],
                    marker=sty["mk"],
                    ms=sty["ms"],
                    label=A_MAIN_DISPLAY[name],
                )
            )
        packed_groups.append((title, hs))
    n_rows = 1 + max(len(hs) for _, hs in packed_groups)
    blank = Line2D([], [], linestyle="none", marker="none", color="none", label=" ")
    packed: list = []
    for title, hs in packed_groups:
        col = [_Hdr(title)] + list(hs)
        while len(col) < n_rows:
            col.append(blank)
        packed.extend(col)
    leg = fig.legend(
        handles=packed,
        loc="outside lower center",
        ncol=3,
        frameon=False,
        fontsize=7,
        handlelength=2.2,
        columnspacing=1.8,
        handletextpad=0.45,
        labelspacing=0.35,
        handler_map={_Hdr: _NoHandle()},
        alignment="left",
    )
    headers = {"No weight correction", "Weight correction", "Proposed"}
    proposed_labs = {A_MAIN_DISPLAY[n] for n in A_MAIN_PROP}
    for t in leg.get_texts():
        txt = t.get_text()
        if txt in headers:
            t.set_fontstyle("italic")
        elif txt in proposed_labs:
            t.set_fontweight("bold")
    save_png(fig, out_path)


def _b_style(mid: str) -> dict:
    name = B_MID_STYLE_NAME[mid]
    sty = SERIES_STYLE[name]
    return dict(color=sty["color"], mk=sty["mk"])


def plot_b_main(cells: dict, out_path: Path) -> None:
    apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(FULL_W, 6.85), layout="constrained", sharey="row")
    y = {mid: float(i) for i, mid in enumerate(B_MAIN_ORDER)}
    letters = "abcd"
    group_breaks = (("RAFR", "HDR_W"), ("DSCP", "HDR_PROP"))
    for i, (row_i, cal, kind) in enumerate(
        ((0, "TinyFace", "R"), (0, "TinyFace", "C"), (1, "XQLFW", "R"), (1, "XQLFW", "C"))
    ):
        ax = axes[row_i][0 if kind == "R" else 1]
        for a, b in group_breaks:
            ax.axhline(0.5 * (y[a] + y[b]), color="0.65", lw=0.55, zorder=1, solid_capstyle="butt")
        for mid in B_MAIN_ORDER:
            if mid in B_HEADERS:
                continue
            rec = cells.get(mid, {}).get(cal)
            if rec is None or not np.isfinite(rec["c"]):
                continue
            sty = _b_style(mid)
            yy = y[mid]
            mk_kw = dict(
                marker=sty["mk"],
                facecolors=sty["color"],
                edgecolors=sty["color"],
                linewidths=0.9,
                zorder=4,
                clip_on=False,
            )
            if kind == "R":
                x = rec["r"]
                if not np.isfinite(x):
                    continue
                lo, hi = rec["lo"], rec["hi"]
                if np.isfinite(lo) and np.isfinite(hi):
                    ax.errorbar(
                        x,
                        yy,
                        xerr=np.array([[max(0.0, x - lo)], [max(0.0, hi - x)]]),
                        fmt="none",
                        ecolor=sty["color"],
                        elinewidth=1.15,
                        capsize=2.2,
                        capthick=0.9,
                        zorder=3,
                        clip_on=False,
                    )
                ax.scatter([x], [yy], s=28 if sty["mk"] not in {"P", "*"} else 36, **mk_kw)
            else:
                x = max(rec["c"], 1e-3)
                ax.scatter([x], [yy], s=28 if sty["mk"] not in {"P", "*"} else 36, **mk_kw)
        labels = []
        for mid in B_MAIN_ORDER:
            lab = B_MAIN_DISPLAY[mid]
            if cal == "XQLFW" and mid == "WACR_PAC":
                lab = r"WACR-PAC ($\Gamma = 1$)$^{\ddagger}$"
            labels.append(lab)
        ax.set_yticks(list(y.values()))
        ax.set_yticklabels(labels, fontsize=6.0)
        for tick, mid in zip(ax.get_yticklabels(), B_MAIN_ORDER):
            if mid in B_HEADERS:
                tick.set_fontstyle("italic")
            elif mid in B_PROPOSED:
                tick.set_fontweight("bold")
        for ytick, mid in zip(ax.yaxis.get_major_ticks(), B_MAIN_ORDER):
            if mid in B_HEADERS:
                ytick.tick1line.set_visible(False)
                ytick.tick2line.set_visible(False)
        ax.set_ylim(-0.55, float(len(B_MAIN_ORDER) - 1) + 0.55)
        ax.grid(True, axis="x", alpha=0.25, linestyle="--", linewidth=0.4)
        ax.set_axisbelow(True)
        if kind == "R":
            ax.axvline(HEAD, color="black", ls="--", lw=0.8, zorder=2)
            ax.set_xlim(-0.018, 0.35)
            if row_i == 1:
                ax.set_xlabel(r"$\hat{R}$")
        else:
            ax.set_xscale("log")
            ax.set_xlim(1.0, 1000.0)
            ax.xaxis.set_major_locator(LogLocator(base=10.0, numticks=5))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.axvline(585.0, color="0.4", ls=":", lw=0.8, zorder=2)
            ax.text(
                585.0,
                1.02,
                "full gallery",
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=5.5,
                color="0.4",
                clip_on=False,
                zorder=5,
            )
            if row_i == 1:
                ax.set_xlabel(r"mean $|C|$")
        ax.set_title(f"({letters[i]}) {cal} " + r"$\to$ SCFace", loc="left", fontsize=8)
    save_png(fig, out_path)


def _sens_style_log_x(ax) -> None:
    ax.set_xscale("log")
    ax.set_xlim(SENS_G_LO, SENS_G_HI)
    ax.set_xticks([1.0, 1.5, 2.0, 3.0, 5.0])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.4)
    ax.set_axisbelow(True)


def _sens_curves_raw(rows, cal, meth, alpha):
    want = f"{float(alpha):.2f}"
    recs = []
    for r in rows:
        if r.get("kind") != "curve" or r.get("cal") != cal or r.get("method") != meth:
            continue
        if f"{_f(r['alpha']):.2f}" != want:
            continue
        g = _f(r["gamma"])
        if g > SENS_G_HI + 1e-9:
            continue
        recs.append(
            (
                g,
                _f(r["r_hat"]),
                _f(r["strat_lo"]) if r.get("strat_lo") not in ("", None) else np.nan,
                _f(r["strat_hi"]) if r.get("strat_hi") not in ("", None) else np.nan,
                _f(r["set_size"]),
            )
        )
    recs.sort(key=lambda t: t[0])
    if not recs:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty, empty, empty, empty
    xs, rh, lo, hi, cs = zip(*recs)
    return (
        np.asarray(xs, dtype=np.float64),
        np.asarray(rh, dtype=np.float64),
        np.asarray(lo, dtype=np.float64),
        np.asarray(hi, dtype=np.float64),
        np.asarray(cs, dtype=np.float64),
    )


def _sens_curves(rows, cal, meth, alpha):
    want = f"{alpha:.2f}"
    xs, rh, lo, hi, cs = [], [], [], [], []
    for r in rows:
        if r["kind"] != "curve" or r["cal"] != cal or r["method"] != meth:
            continue
        if r["alpha"] != want:
            continue
        g = _f(r["gamma"])
        if g > SENS_G_HI + 1e-9:
            continue
        xs.append(g)
        rh.append(_f(r["r_hat"]))
        lo.append(_f(r["strat_lo"]) if r["strat_lo"] not in ("", None) else np.nan)
        hi.append(_f(r["strat_hi"]) if r["strat_hi"] not in ("", None) else np.nan)
        cs.append(_f(r["set_size"]))
    order = np.argsort(xs)
    xs = np.asarray(xs, dtype=np.float64)[order]
    return (
        xs,
        np.asarray(rh, dtype=np.float64)[order],
        np.asarray(lo, dtype=np.float64)[order],
        np.asarray(hi, dtype=np.float64)[order],
        np.asarray(cs, dtype=np.float64)[order],
    )


def _sens_value(rows, cal, meth, alpha) -> dict:
    want = f"{alpha:.2f}"
    for r in rows:
        if r["kind"] == "value" and r["cal"] == cal and r["method"] == meth and r["alpha"] == want:
            return r
    raise SystemExit(f"missing value {cal} {meth} a={want}")


def _hat_tick(ax, g: float, color: str) -> None:
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.plot(
        [g, g],
        [-0.03, 0.10],
        transform=trans,
        color=color,
        lw=1.8,
        solid_capstyle="butt",
        zorder=5,
        clip_on=False,
    )


def _row2_ylim(rows: list[dict]) -> tuple[float, float]:
    ymin, ymax = 1e9, float(SENS_N_GAL)
    for _comb, cal in SENS_PAIRS:
        for a in SENS_ALPHAS:
            _g, _rh, _lo, _hi, cs = _sens_curves(rows, cal, "B", a)
            ok = np.isfinite(cs) & (cs > 0)
            if np.any(ok):
                ymin = min(ymin, float(np.min(cs[ok])))
                ymax = max(ymax, float(np.max(cs[ok])))
    return max(1.8, ymin * 0.80), max(SENS_N_GAL * 1.18, ymax * 1.05)


def plot_sensitivity(rows: list[dict], png_path: Path, pdf_path: Path) -> None:
    apply_style()
    fig, axes = plt.subplots(
        2, 2, figsize=(FULL_W, 5.20), layout="constrained", sharex=True
    )
    arrow = "\u2192"
    tags = {
        (0, 0): f"(a) TinyFace {arrow} SCFace",
        (0, 1): f"(b) XQLFW {arrow} SCFace",
        (1, 0): f"(c) TinyFace {arrow} SCFace, Method B",
        (1, 1): f"(d) XQLFW {arrow} SCFace, Method B",
    }
    y0, y1 = _row2_ylim(rows)
    for c_i, (_comb, cal) in enumerate(SENS_PAIRS):
        ax = axes[0][c_i]
        n = SENS_N_CAL[cal]
        gvac10 = gamma_vac(SENS_HEAD_A, n, SENS_B_CAP)
        rmax = SENS_HEAD_A
        for meth in ("A", "B"):
            g, rh, lo, hi, _cs = _sens_curves(rows, cal, meth, SENS_HEAD_A)
            eg, erh, elo, ehi, ecs = _sens_curves_raw(rows, cal, meth, SENS_HEAD_A)
            _arr_eq(f"sens {cal} {meth} gamma", g, eg)
            _arr_eq(f"sens {cal} {meth} rhat", rh, erh)
            _arr_eq(f"sens {cal} {meth} lo", lo, elo)
            _arr_eq(f"sens {cal} {meth} hi", hi, ehi)
            _arr_eq(f"sens {cal} {meth} C", _cs, ecs)
            c = SCORE_COLOR[meth]
            ax.fill_between(g, lo, hi, color=c, alpha=0.20, linewidth=0, zorder=2)
            ax.plot(
                g,
                rh,
                color=c,
                marker=SCORE_MK[meth],
                ms=3.4,
                lw=1.4,
                ls=SCORE_LS[meth],
                zorder=3,
            )
            rmax = max(rmax, float(np.nanmax(hi)), float(np.nanmax(rh)))
            rec = _sens_value(rows, cal, meth, SENS_HEAD_A)
            ghat, gcons = _f(rec["gamma_hat"]), _f(rec["gamma_cons"])
            if np.isfinite(ghat):
                _hat_tick(ax, ghat, c)
            if np.isfinite(gcons):
                ax.axvline(gcons, color=c, ls=":", lw=1.15, zorder=2)
        ymax = max(0.12, rmax * 1.08)
        ax.axvspan(gvac10, SENS_G_HI, color="0.75", alpha=0.28, lw=0, zorder=0)
        ax.axhline(SENS_HEAD_A, color="black", ls="--", lw=0.85, zorder=1)
        ax.text(
            min(4.55, 0.5 * (gvac10 + SENS_G_HI)),
            0.28 * ymax,
            "whole gallery",
            ha="center",
            va="center",
            fontsize=7,
            color="0.30",
            zorder=4,
        )
        ax.set_ylim(0.0, ymax)
        if c_i == 0:
            ax.set_ylabel(r"$\hat{R}$")
        ax.set_title(tags[(0, c_i)], loc="left", fontsize=8)
        _sens_style_log_x(ax)

        axb = axes[1][c_i]
        for a in SENS_ALPHAS:
            g, _rh, _lo, _hi, cs = _sens_curves(rows, cal, "B", a)
            eg, _erh, _elo, _ehi, ecs = _sens_curves_raw(rows, cal, "B", a)
            _arr_eq(f"sens {cal} B a={a} C", cs, ecs)
            _arr_eq(f"sens {cal} B a={a} g", g, eg)
            axb.plot(
                g,
                cs,
                color=ROW2_COLOR[a],
                marker=ROW2_MK[a],
                ms=3.4,
                lw=1.4,
                zorder=3,
            )
            axb.axvline(
                gamma_vac(a, n, SENS_B_CAP),
                color=ROW2_COLOR[a],
                ls="--",
                lw=1.05,
                zorder=1,
            )
        axb.axhline(SENS_N_GAL, color="0.40", ls="-.", lw=0.9, zorder=2)
        axb.set_yscale("log")
        axb.set_ylim(y0, y1)
        ax2 = axb.twinx()
        ax2.set_yscale("log")
        ax2.set_ylim(100.0 * y0 / SENS_N_GAL, 100.0 * y1 / SENS_N_GAL)
        ax2.tick_params(axis="y", labelsize=7)
        if c_i == 1:
            ax2.set_ylabel(r"$|C|$ / 585 (\%)")
        else:
            ax2.tick_params(labelright=False)
        if c_i == 0:
            axb.set_ylabel(r"mean $|C|$")
        axb.set_xlabel(r"sensitivity $\Gamma$")
        axb.set_title(tags[(1, c_i)], loc="left", fontsize=8)
        _sens_style_log_x(axb)

    for ax in axes[0]:
        ax.tick_params(labelbottom=False)
    handles_top = [
        Line2D(
            [0],
            [0],
            color=SCORE_COLOR[m],
            marker=SCORE_MK[m],
            ms=4,
            lw=1.4,
            ls=SCORE_LS[m],
            label=f"Method {m}",
        )
        for m in ("A", "B")
    ] + [
        Line2D([0], [0], color="black", ls="--", lw=0.85, label=r"$\alpha=0.10$"),
        Line2D(
            [0],
            [0],
            color=SCORE_COLOR["B"],
            ls=":",
            lw=1.15,
            label=r"conservative $\Gamma$ (dotted, per method)",
        ),
        Line2D([0], [0], color="0.2", lw=1.8, label=r"$\hat\Gamma^{\ast}$ (point)"),
    ]
    axes[0][1].legend(
        handles=handles_top,
        loc="upper right",
        frameon=False,
        fontsize=7,
    )
    handles_bot = [
        Line2D(
            [0],
            [0],
            color=ROW2_COLOR[a],
            marker=ROW2_MK[a],
            ms=4,
            lw=1.4,
            label=rf"$\alpha={a:.2f}$",
        )
        for a in SENS_ALPHAS
    ] + [
        Line2D(
            [0],
            [0],
            color="0.40",
            ls="-.",
            lw=0.9,
            label=r"whole gallery ($|C|=585$)",
        ),
    ]
    fig.legend(
        handles=handles_bot,
        loc="outside lower center",
        ncol=4,
        frameon=False,
        fontsize=7,
    )
    save_pdf_png(fig, pdf_path, png_path, dpi=400)


def binned_median(t: np.ndarray, y: np.ndarray, n_bins: int = ADAPT_N_BINS):
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(t, qs))
    if edges.size < 3:
        raise SystemExit("too few unique t values for 20 equal-count bins")
    centres = []
    meds = []
    idx = np.digitize(t, edges[1:-1], right=True)
    for b in range(edges.size - 1):
        m = idx == b
        if int(m.sum()) == 0:
            continue
        centres.append(float(np.median(t[m])))
        meds.append(float(np.median(y[m])))
    return np.asarray(centres), np.asarray(meds)


def y_display(size: np.ndarray) -> np.ndarray:
    out = size.astype(np.float64)
    out[out < 1] = 0.85
    return out


def _adapt_panels(rows: list[dict]) -> list[dict]:
    panels = []
    for key in ("a", "b", "c"):
        spec = ADAPT_SPEC[key]
        sub = [r for r in rows if r["panel"] == key]
        if len(sub) != ADAPT_N_PROBE:
            raise SystemExit(f"adaptivity panel {key}: {len(sub)} rows, expected {ADAPT_N_PROBE}")
        t = np.array([_f(r["t"]) for r in sub], dtype=np.float64)
        size = np.array([int(float(r["set_size"])) for r in sub], dtype=np.int64)
        hit = np.array([int(float(r["hit"])) for r in sub], dtype=bool)
        dist = np.array([int(float(r["distance_code"])) for r in sub], dtype=np.int64)
        miss_by_dist = [float((~hit[dist == d]).mean()) for d in DIST_ORDER]
        panels.append(
            {
                "spec": spec,
                "t": t,
                "size": size,
                "hit": hit,
                "dist": dist,
                "r_hat": float((~hit).mean()),
                "mean_c": float(size.mean()),
                "miss_by_dist": miss_by_dist,
            }
        )
    return panels


def plot_adaptivity(rows: list[dict], png_path: Path, pdf_path: Path) -> None:
    panels = _adapt_panels(rows)
    apply_style()
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(FULL_W, 4.55),
        layout="constrained",
        sharey="row",
    )
    y_lo, y_hi = 1.0, ADAPT_Y_HI
    has_zero = any(int(p["size"].min()) == 0 for p in panels)
    if has_zero:
        y_lo = 0.80

    for col, pan in enumerate(panels):
        ax = axes[0, col]
        t, size, hit = pan["t"], pan["size"], pan["hit"]
        yd = y_display(size)
        raw_t = np.array(
            [_f(r["t"]) for r in rows if r["panel"] == pan["spec"]["key"]],
            dtype=np.float64,
        )
        raw_size = np.array(
            [int(float(r["set_size"])) for r in rows if r["panel"] == pan["spec"]["key"]],
            dtype=np.int64,
        )
        raw_hit = np.array(
            [int(float(r["hit"])) for r in rows if r["panel"] == pan["spec"]["key"]],
            dtype=bool,
        )
        _arr_eq(f"adapt {pan['spec']['key']} t", t, raw_t)
        _arr_eq(f"adapt {pan['spec']['key']} size", size, raw_size)
        _arr_eq(f"adapt {pan['spec']['key']} hit", hit.astype(np.float64), raw_hit.astype(np.float64))
        ax.scatter(
            t[hit],
            yd[hit],
            s=8,
            c=COLOR_HIT,
            alpha=0.70,
            linewidths=0,
            zorder=3,
            rasterized=True,
        )
        ax.scatter(
            t[~hit],
            yd[~hit],
            s=16,
            c=COLOR_MISS,
            alpha=0.90,
            linewidths=0,
            zorder=4,
            rasterized=True,
        )
        xc, yc = binned_median(t, size.astype(np.float64))
        ax.plot(xc, np.maximum(yc, 0.85), color=COLOR_MEDIAN, lw=1.5, zorder=5)
        ax.axhline(ADAPT_N_GAL, color="0.35", ls="--", lw=0.9, zorder=2)
        ax.set_yscale("log")
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel(r"Top-1 similarity $t$")
        if col == 0:
            ax.set_ylabel(r"Set size $|C(q)|$")
        ax.grid(True, which="both", alpha=0.22, ls="--", lw=0.35)
        ax.set_title(pan["spec"]["title"], loc="left", fontsize=8, pad=3)
        ax.text(
            0.97,
            0.03,
            rf"$\hat R={pan['r_hat']:.3f}$" + "\n" + rf"mean $|C|={pan['mean_c']:.1f}$",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
        )

        bx = axes[1, col]
        groups = [y_display(size[pan["dist"] == d]) for d in DIST_ORDER]
        raw_dist = np.array(
            [int(float(r["distance_code"])) for r in rows if r["panel"] == pan["spec"]["key"]],
            dtype=np.int64,
        )
        for i, d in enumerate(DIST_ORDER):
            exp = y_display(raw_size[raw_dist == d])
            _arr_eq(f"adapt {pan['spec']['key']} box d={d}", groups[i], exp)
        bp = bx.boxplot(
            groups,
            positions=[1, 2, 3],
            widths=0.55,
            patch_artist=True,
            whis=1.5,
            showfliers=False,
            medianprops={"color": COLOR_MEDIAN, "linewidth": 1.4},
            whiskerprops={"color": "0.25", "linewidth": 0.7},
            capprops={"color": "0.25", "linewidth": 0.7},
            boxprops={"facecolor": "#DDEAF3", "edgecolor": "0.25", "linewidth": 0.7},
        )
        del bp
        bx.set_yscale("log")
        bx.set_ylim(y_lo, y_hi)
        bx.set_xticks([1, 2, 3], DIST_LABELS)
        bx.set_xlabel("Capture distance")
        if col == 0:
            bx.set_ylabel(r"Set size $|C(q)|$")
        bx.axhline(ADAPT_N_GAL, color="0.35", ls="--", lw=0.9, zorder=1)
        bx.grid(True, which="both", axis="y", alpha=0.22, ls="--", lw=0.35)
        bx.set_title(pan["spec"]["title_d"], loc="left", fontsize=8, pad=3)
        for i, d in enumerate(DIST_ORDER, start=1):
            mr = pan["miss_by_dist"][i - 1]
            ymax = float(np.percentile(size[pan["dist"] == d], 95))
            y_lab = min(ymax * 1.35, 420.0)
            bx.text(
                i,
                max(y_lab, 12.0),
                f"{100.0 * mr:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7,
                color=COLOR_MISS,
            )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLOR_HIT,
            markersize=5,
            label="Hit",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=COLOR_MISS,
            markersize=7,
            label="Miss",
        ),
        Line2D([0], [0], color=COLOR_MEDIAN, lw=1.5, label="Binned median"),
        Line2D([0], [0], color="0.35", ls="--", lw=0.9, label=r"Whole gallery ($|G|=585$)"),
    ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=4,
        frameon=False,
        fontsize=7,
    )
    save_pdf_png(fig, pdf_path, png_path, dpi=600)


def _wq_summary(rows: list[dict], cal: str, weight: str) -> dict | None:
    pair = f"{cal} -> SCFace"
    for r in rows:
        if r["kind"] == "summary" and r["pair"] == pair and r["weight"] == weight:
            return r
    return None


def _wq_stairs(rows: list[dict], pair: str, weight: str, kind: str):
    lo, hi, dens = [], [], []
    for r in rows:
        if r["kind"] != kind or r["pair"] != pair or r["weight"] != weight:
            continue
        lo.append(_f(r["t_lo"]))
        hi.append(_f(r["t_hi"]))
        dens.append(_f(r["density"]))
    if not lo:
        return None, None
    edges = np.append(np.array(lo, dtype=np.float64), float(hi[-1]))
    return edges, np.array(dens, dtype=np.float64)


def _wq_stairs_raw(rows: list[dict], pair: str, weight: str, kind: str):
    recs = [
        (_f(r["t_lo"]), _f(r["t_hi"]), _f(r["density"]))
        for r in rows
        if r.get("kind") == kind and r.get("pair") == pair and r.get("weight") == weight
    ]
    recs.sort(key=lambda t: t[0])
    if not recs:
        return None, None
    lo, hi, dens = zip(*recs)
    edges = np.append(np.array(lo, dtype=np.float64), float(hi[-1]))
    return edges, np.array(dens, dtype=np.float64)


def _tv_label(rows: list[dict], cal: str, name: str) -> str:
    shown = WEIGHT_LEGEND.get(name, name)
    rec = _wq_summary(rows, cal, name)
    if rec is None or rec.get("tv") in ("", None):
        return shown
    return f"{shown} (TV = {_f(rec['tv']):.3f})"


def plot_weight_quality(rows: list[dict], png_path: Path, pdf_path: Path) -> None:
    apply_style()
    fig, axes = plt.subplots(
        2, 2, figsize=(FULL_W, 5.05), layout="constrained", sharey=True
    )
    specs = (
        (0, 0, "TinyFace", WEIGHT_EMBED, r"(a) TinyFace $\to$ SCFace, embedding-level weights"),
        (0, 1, "TinyFace", WEIGHT_ON_T, r"(b) TinyFace $\to$ SCFace, weights on $t$"),
        (1, 0, "XQLFW", WEIGHT_EMBED, r"(c) XQLFW $\to$ SCFace, embedding-level weights"),
        (1, 1, "XQLFW", WEIGHT_ON_T, r"(d) XQLFW $\to$ SCFace, weights on $t$"),
    )
    ymax = 0.0
    for r_i, c_i, cal, names, title in specs:
        ax = axes[r_i][c_i]
        pair = f"{cal} -> SCFace"
        e_te, d_te = _wq_stairs(rows, pair, "SCFace", "target")
        er, dr = _wq_stairs_raw(rows, pair, "SCFace", "target")
        if e_te is not None:
            _arr_eq(f"wq {cal} target edges", e_te, er)
            _arr_eq(f"wq {cal} target dens", d_te, dr)
            ax.stairs(
                d_te,
                e_te,
                fill=True,
                color="0.80",
                alpha=0.42,
                baseline=0.0,
                zorder=0,
                label="SCFace (target)",
            )
            ymax = max(ymax, float(np.nanmax(d_te)))
        rec0 = _wq_summary(rows, cal, "Unweighted") or _wq_summary(rows, cal, "WACR (histogram)")
        if rec0 is not None and rec0.get("t_cut") not in ("", None):
            ax.axvline(_f(rec0["t_cut"]), color="0.35", ls="--", lw=1.0, zorder=1)
        for name in names:
            e, d = _wq_stairs(rows, pair, name, "density")
            if e is None:
                continue
            er, dr = _wq_stairs_raw(rows, pair, name, "density")
            _arr_eq(f"wq {cal} {name} edges", e, er)
            _arr_eq(f"wq {cal} {name} dens", d, dr)
            ymax = max(ymax, float(np.nanmax(d)))
            sty = WEIGHT_STAIRS[name]
            ax.stairs(
                d,
                e,
                fill=False,
                color=sty["color"],
                linestyle=sty["ls"],
                linewidth=sty["lw"],
                baseline=0.0,
                zorder=3,
                label=_tv_label(rows, cal, name),
            )
        ax.set_title(title, loc="left", fontsize=8)
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.4)
        ax.set_axisbelow(True)
        ax.set_xlim(0.0, 0.88)
        ax.legend(frameon=False, loc="upper right", fontsize=7, handlelength=2.4)
        if r_i == 1:
            ax.set_xlabel(r"top-1 cosine $t$")
        if c_i == 0:
            ax.set_ylabel("Density")
    ytop = float(np.ceil(ymax * 1.02))
    for ax in axes.ravel():
        ax.set_ylim(0.0, ytop)
        ax.yaxis.set_major_locator(MultipleLocator(1))
    save_pdf_png(fig, pdf_path, png_path, dpi=400)


def _cond_xy(rows, panel, domain, kind):
    xs, ys, los, his = [], [], [], []
    for r in rows:
        if r["kind"] != kind or r["panel"] != panel or r["domain"] != domain:
            continue
        if r["t_mean"] in ("", None) or r["p"] in ("", None):
            continue
        xs.append(_f(r["t_mean"]))
        ys.append(_f(r["p"]))
        los.append(_f(r["lo"]) if r["lo"] not in ("", None) else np.nan)
        his.append(_f(r["hi"]) if r["hi"] not in ("", None) else np.nan)
    return np.array(xs), np.array(ys), np.array(los), np.array(his)


def _cond_xy_raw(rows, panel, domain, kind):
    recs = []
    for r in rows:
        if r.get("kind") != kind or r.get("panel") != panel or r.get("domain") != domain:
            continue
        if r.get("t_mean") in ("", None) or r.get("p") in ("", None):
            continue
        recs.append(
            (
                _f(r["t_mean"]),
                _f(r["p"]),
                _f(r["lo"]) if r.get("lo") not in ("", None) else np.nan,
                _f(r["hi"]) if r.get("hi") not in ("", None) else np.nan,
            )
        )
    if not recs:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty, empty, empty
    xs, ys, los, his = zip(*recs)
    return (
        np.asarray(xs, dtype=np.float64),
        np.asarray(ys, dtype=np.float64),
        np.asarray(los, dtype=np.float64),
        np.asarray(his, dtype=np.float64),
    )


def _draw_band(ax, x, y, lo, hi, name):
    c, mk = DS_COLOR[name], DS_MK[name]
    ok = np.isfinite(x) & np.isfinite(y)
    if not np.any(ok):
        return
    band = np.isfinite(lo) & np.isfinite(hi) & ok
    if np.any(band):
        ax.fill_between(x[band], lo[band], hi[band], color=c, alpha=0.22, linewidth=0, zorder=2)
    ax.plot(x[ok], y[ok], color=c, marker=mk, ms=4.5, lw=1.4, zorder=3, label=name)


def _cond_density(ax, rows, panel):
    lo, hi, dens = [], [], []
    for r in rows:
        if r["kind"] != "density" or r["panel"] != panel:
            continue
        lo.append(_f(r["t_lo"]))
        hi.append(_f(r["t_hi"]))
        dens.append(_f(r["p"]))
    if not lo:
        return None
    raw = [
        (_f(r["t_lo"]), _f(r["t_hi"]), _f(r["p"]))
        for r in rows
        if r.get("kind") == "density" and r.get("panel") == panel
    ]
    rlo, rhi, rd = zip(*raw)
    _arr_eq(f"cond dens {panel} lo", lo, rlo)
    _arr_eq(f"cond dens {panel} hi", hi, rhi)
    _arr_eq(f"cond dens {panel} p", dens, rd)
    lo, hi, dens = np.array(lo), np.array(hi), np.array(dens)
    x = np.append(lo, hi[-1])
    y = np.append(dens, dens[-1])
    ax2 = ax.twinx()
    ax2.fill_between(x, y, step="post", color="0.80", alpha=0.35, linewidth=0, zorder=0)
    ax2.set_ylabel("SCFace density", color="0.45", fontsize=7)
    ax2.tick_params(axis="y", labelsize=7, colors="0.45")
    ax2.set_ylim(0.0, max(float(np.nanmax(dens)) * 3.2, 1e-6))
    ax2.set_zorder(0)
    ax.set_zorder(1)
    ax.patch.set_visible(False)
    return ax2


def _cond_summary(rows, cal):
    pair = f"{cal} -> SCFace"
    for r in rows:
        if r["kind"] == "summary" and r["pair"] == pair:
            return r
    raise SystemExit(f"missing summary {pair}")


def plot_conditional(rows: list[dict], png_path: Path, pdf_path: Path) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 2.85), layout="constrained")
    ax_a, ax_b, ax_c = axes

    for name in ("TinyFace", "XQLFW", "SCFace"):
        x, y, lo, hi = _cond_xy(rows, "a", name, "hit")
        xr, yr, lor, hir = _cond_xy_raw(rows, "a", name, "hit")
        _arr_eq(f"cond hit {name} x", x, xr)
        _arr_eq(f"cond hit {name} y", y, yr)
        _arr_eq(f"cond hit {name} lo", lo, lor)
        _arr_eq(f"cond hit {name} hi", hi, hir)
        _draw_band(ax_a, x, y, lo, hi, name)
    ax_a.set_ylim(0.0, 1.05)
    ax_a.set_ylabel(r"$P($rank-1 hit$\mid t)$")
    ax_a.set_xlabel(r"top-1 cosine $t$")
    ax_a.set_title(r"(a) Rank-1 hit given $t$", loc="left", fontsize=8)
    ax_a.legend(frameon=False, loc="lower right", fontsize=7)
    ax_a.grid(True, alpha=0.25, linestyle="--", linewidth=0.4)
    ax_a.set_axisbelow(True)
    ax_a.set_xlim(0.0, 0.85)

    for ax, panel, src, tag, dxy, dha, leg_loc in (
        (ax_b, "b", "TinyFace", r"(b) TinyFace $\to$ SCFace", (0.04, 0.96), "left", "upper right"),
        (ax_c, "c", "XQLFW", r"(c) XQLFW $\to$ SCFace", (0.97, 0.96), "right", "center right"),
    ):
        _cond_density(ax, rows, panel)
        x, y, lo, hi = _cond_xy(rows, panel, src, "exc")
        xr, yr, lor, hir = _cond_xy_raw(rows, panel, src, "exc")
        _arr_eq(f"cond exc {src} x", x, xr)
        _arr_eq(f"cond exc {src} y", y, yr)
        _arr_eq(f"cond exc {src} lo", lo, lor)
        _arr_eq(f"cond exc {src} hi", hi, hir)
        _draw_band(ax, x, y, lo, hi, src)
        x, y, lo, hi = _cond_xy(rows, panel, "SCFace", "exc")
        xr, yr, lor, hir = _cond_xy_raw(rows, panel, "SCFace", "exc")
        _arr_eq(f"cond exc SCFace {panel} x", x, xr)
        _arr_eq(f"cond exc SCFace {panel} y", y, yr)
        _draw_band(ax, x, y, lo, hi, "SCFace")
        rec = _cond_summary(rows, src)
        ax.text(
            dxy[0],
            dxy[1],
            rf"$\Delta={_f(rec['delta']):+.3f}$",
            transform=ax.transAxes,
            va="top",
            ha=dha,
            fontsize=8,
            zorder=5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", alpha=0.92),
        )
        ax.set_ylim(0.0, 0.85)
        ax.set_xlim(0.0, 0.85)
        ax.set_xlabel(r"top-1 cosine $t$")
        ax.set_title(tag, loc="left", fontsize=8)
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.4)
        ax.set_axisbelow(True)
        handles = [
            Line2D(
                [0],
                [0],
                color=DS_COLOR[src],
                marker=DS_MK[src],
                ls="-",
                lw=1.4,
                ms=4.5,
                label=src,
            ),
            Line2D(
                [0],
                [0],
                color=DS_COLOR["SCFace"],
                marker=DS_MK["SCFace"],
                ls="-",
                lw=1.4,
                ms=4.5,
                label="SCFace",
            ),
        ]
        ax.legend(handles=handles, frameon=False, loc=leg_loc, fontsize=7)
    ax_b.set_ylabel(r"$P(S>\hat{\tau}\mid t)$")
    save_pdf_png(fig, pdf_path, png_path, dpi=400)


def write_compare_sheet(pairs: list[tuple[str, Path | None, Path | None]], dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    n = len(pairs)
    fig, axes = plt.subplots(n, 2, figsize=(10.5, 2.6 * n), layout="constrained")
    if n == 1:
        axes = np.array([axes])
    for i, (title, orig, redraw) in enumerate(pairs):
        for j, (lab, path) in enumerate((("original", orig), ("WACR redraw", redraw))):
            ax = axes[i][j]
            ax.set_axis_off()
            p = None if path is None else Path(path)
            if p is None or not p.is_file():
                ax.set_title(f"{title}: {lab} missing", loc="left", fontsize=8)
                continue
            img = plt.imread(str(p))
            ax.imshow(img)
            ax.set_title(f"{title}: {lab}", loc="left", fontsize=8)
    fig.savefig(dest, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {dest}", flush=True)


PAPER_COMPARE = (
    ("Fig 4 sensitivity", "Fig06_sensitivity.png", "fig_sensitivity.png"),
    ("Fig 6 adaptivity", "FigC4_adaptivity.png", "fig_adaptivity.png"),
    ("Fig 7 weight quality", "Fig04_weight_quality.png", "fig_weight_quality.png"),
    ("Fig 8 conditional", "Fig05_a1_diagnostic.png", "fig_conditional.png"),
)


def _find_paper_png(root: Path, name: str) -> Path | None:
    direct = root / name
    if direct.is_file():
        return direct
    hits = sorted(root.rglob(name))
    return hits[0] if hits else None


def write_paper_compare(redraw_dir: Path, dest: Path) -> None:
    root_s = os.environ.get("WACR_PAPER_FIGS", "").strip()
    root = Path(root_s) if root_s else None
    pairs: list[tuple[str, Path | None, Path | None]] = []
    for title, orig_name, redraw_name in PAPER_COMPARE:
        orig = None
        if root is not None and root.is_dir():
            orig = _find_paper_png(root, orig_name)
        redraw = redraw_dir / redraw_name
        pairs.append((title, orig, redraw if redraw.is_file() else None))
    write_compare_sheet(pairs, dest)

