"""Benchmark charts (matplotlib), rendered in a light and a dark theme.

Follows the dataviz mark specs: emphasis form (Jev systems in the two validated hues,
baselines gray with distinct marker shapes), 2px lines, >=8px markers with a 2px
surface ring, bars at most 24px thick with a 4px rounded data end and a square
baseline, hairline solid grids, one y-scale per plot, text in ink tokens, a legend
for 2+ series plus direct labels that are dropped rather than allowed to collide.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402
from matplotlib.transforms import Bbox, IdentityTransform  # noqa: E402

THEMES = {
    "light": {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "jevragrank": "#2a78d6",
              "jevrank": "#eb6834", "base": "#898781"},
    "dark": {"surface": "#1a1a19", "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
             "grid": "#2c2c2a", "axis": "#383835", "jevragrank": "#3987e5",
             "jevrank": "#d95926", "base": "#898781"},
}
LABELS = {"bm25": "BM25", "dense": "Dense (bge-base)", "dense+ce": "Dense → cross-encoder",
          "dense+llm": "Dense → Qwen3.5-2B prompt", "jevragrank": "JevRAGRank",
          "jevrank": "JevRank (no embeddings)"}
MARKERS = {"bm25": "s", "dense": "D", "dense+ce": "^", "dense+llm": "v",
           "jevragrank": "o", "jevrank": "o"}
MAIN = ["bm25", "dense", "dense+ce", "dense+llm", "jevragrank", "jevrank"]
# Direct-label priority: the Jev systems always win a contested spot.
LABEL_PRIORITY = ["jevragrank", "jevrank", "dense+ce", "dense", "bm25", "dense+llm"]
DATASET_TITLES = {"scifact": "SciFact", "nfcorpus": "NFCorpus"}

PANEL_W, PANEL_H = 5.2, 4.2  # inches per panel
HEADER_IN = 0.85  # title + subtitle + legend band, inches
HEADER_NO_LEGEND_IN = 0.6  # title + subtitle only
MARKER_PT = 9  # >= 8px marker
BAR_PX, BAR_MAX_PX, BAR_RADIUS_PX = 20.0, 24.0, 4.0
LABEL_PT = 8

# Candidate offsets (points) for a point label: right, left, above, below.
AROUND = [(9, 2, "left", "bottom"), (9, -2, "left", "top"), (-9, 2, "right", "bottom"),
          (-9, -2, "right", "top"), (0, 10, "center", "bottom"), (0, -10, "center", "top"),
          (9, -9, "left", "top"), (-9, -9, "right", "top"), (9, 9, "left", "bottom"),
          (-9, 9, "right", "bottom")]
# Candidate offsets for a line end label: right of the end, then above/below it.
LINE_END = [(10, 0, "left", "center"), (0, 10, "center", "bottom"), (0, -10, "center", "top"),
            (-8, 8, "right", "bottom"), (-8, -8, "right", "top")]


# --------------------------------------------------------------------------- helpers


def _color(t: dict, system: str) -> str:
    return t[system] if system in ("jevragrank", "jevrank") else t["base"]


def _mix(hex_a: str, hex_b: str, w: float) -> tuple:
    a, b = to_rgb(hex_a), to_rgb(hex_b)
    return tuple(w * x + (1 - w) * y for x, y in zip(a, b, strict=True))


def _num(v: float, _pos=None) -> str:
    """Plain tick numbers: 10, 100, 1,000 (never 10^3), 0.5 below one."""
    return f"{v:,.0f}" if abs(v) >= 1 else f"{v:g}"


def _fixed(decimals: int) -> FuncFormatter:
    return FuncFormatter(lambda v, _pos=None: f"{v:.{decimals}f}")


def _log_axis(ax, which: str, values: list[float], pad: float = 1.6) -> None:
    """Log scale with plain-number ticks from the 1-2-5 sequence (decades when wide)."""
    axis = ax.xaxis if which == "x" else ax.yaxis
    (ax.set_xscale if which == "x" else ax.set_yscale)("log")
    vals = [v for v in values if v > 0]
    if not vals:
        return
    lo, hi = min(vals) / pad, max(vals) * pad
    (ax.set_xlim if which == "x" else ax.set_ylim)(lo, hi)
    decades = range(int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi))) + 1)
    ticks = [10.0**d for d in decades if lo <= 10.0**d <= hi]
    if len(ticks) < 3:
        ticks = [m * 10.0**d for d in decades for m in (1, 2, 5) if lo <= m * 10.0**d <= hi]
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FuncFormatter(_num))
    axis.set_minor_locator(NullLocator())


class _Bar(Patch):
    """A bar sized in CSS px at draw time.

    Thickness is capped (<= 24px, and <= 60% of its category band); the data end has a
    4px rounded corner and the baseline end is square. ``offset`` shifts it across the
    band in px, which is how grouped bars get an exact gap between them.
    """

    def __init__(self, ax, pos: float, value: float, *, horizontal: bool,
                 thickness: float = BAR_PX, offset: float = 0.0, base: float = 0.0, **kw):
        super().__init__(linewidth=0, **kw)
        self._ax, self._pos, self._value, self._base = ax, pos, value, base
        self._h, self._thick, self._offset = horizontal, min(thickness, BAR_MAX_PX), offset

    def get_transform(self):
        return IdentityTransform()

    def get_path(self):
        px = self._ax.figure.dpi / 96
        tr = self._ax.transData
        if self._h:
            (b, c), (e, _) = tr.transform([(self._base, self._pos), (self._value, self._pos)])
            ends = tr.transform([(0, self._pos - 0.5), (0, self._pos + 0.5)])[:, 1]
            band = abs(ends[1] - ends[0])
            c -= self._offset * px
        else:
            (c, b), (_, e) = tr.transform([(self._pos, self._base), (self._pos, self._value)])
            ends = tr.transform([(self._pos - 0.5, 0), (self._pos + 0.5, 0)])[:, 0]
            band = abs(ends[1] - ends[0])
            c += self._offset * px
        half = min(self._thick * px, 0.6 * band) / 2
        r = min(BAR_RADIUS_PX * px, half, abs(e - b))
        s = 1.0 if e >= b else -1.0
        k = 0.5523 * r  # cubic Bezier quarter-circle constant
        lo, hi = c - half, c + half
        uw = [(b, lo), (e - s * r, lo), (e - s * r + s * k, lo), (e, lo + r - k), (e, lo + r),
              (e, hi - r), (e, hi - r + k), (e - s * r + s * k, hi), (e - s * r, hi), (b, hi),
              (b, lo)]
        verts = uw if self._h else [(w, u) for u, w in uw]
        codes = [MplPath.MOVETO, MplPath.LINETO, *[MplPath.CURVE4] * 3, MplPath.LINETO,
                 *[MplPath.CURVE4] * 3, MplPath.LINETO, MplPath.CLOSEPOLY]
        return MplPath(verts, codes)


def _bar(ax, pos, value, color, *, horizontal, offset=0.0, thickness=BAR_PX) -> _Bar:
    patch = _Bar(ax, pos, value, horizontal=horizontal, offset=offset, thickness=thickness,
                 facecolor=color)
    ax.add_patch(patch)
    return patch


def _style(ax, t: dict) -> None:
    ax.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["axis"])
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=t["muted"], labelcolor=t["ink2"], labelsize=9, length=0, pad=5)
    ax.grid(True, color=t["grid"], linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(t["ink2"])
    ax.yaxis.label.set_color(t["ink2"])
    ax.title.set_color(t["ink"])


def _figure(t: dict, ncols: int, title: str, subtitle: str, height: float = PANEL_H):
    ncols = max(1, ncols)
    fig, axes = plt.subplots(1, ncols, figsize=(PANEL_W * ncols, height), squeeze=False)
    fig.patch.set_facecolor(t["surface"])
    fig.text(0.1 / (PANEL_W * ncols), 1 - 0.12 / height, title, ha="left", va="top",
             fontsize=13, fontweight="semibold", color=t["ink"])
    fig.text(0.1 / (PANEL_W * ncols), 1 - 0.40 / height, subtitle, ha="left", va="top",
             fontsize=9, color=t["ink2"])
    fig._jrr_labels = []  # deferred direct labels, placed once the layout is final
    fig._jrr_header = HEADER_NO_LEGEND_IN  # _fig_legend grows it when a legend row is added
    for ax in axes[0]:
        _style(ax, t)
    return fig, list(axes[0])


def _fig_legend(fig, t: dict, handles: list, labels: list[str]) -> None:
    """One legend row under the subtitle, shared by every panel."""
    if len(handles) < 2:
        return
    w, h = fig.get_size_inches()
    leg = fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.1 / w, 1 - 0.60 / h),
                     ncol=len(handles), frameon=False, fontsize=LABEL_PT, handlelength=1.6,
                     handletextpad=0.5, columnspacing=1.8, borderaxespad=0, borderpad=0)
    for text in leg.get_texts():
        text.set_color(t["ink2"])
    fig._jrr_header = HEADER_IN


def _marker_handle(t: dict, s: str, line: bool = False) -> Line2D:
    return Line2D([], [], color=_color(t, s), marker=MARKERS[s], ms=MARKER_PT - 1,
                  mec=t["surface"], mew=1.5, lw=2 if line else 0)


def _defer_labels(fig, ax, t: dict, entries: list[tuple[str, tuple, list]]) -> None:
    """Queue direct labels (text, data xy, candidate offsets) in priority order."""
    fig._jrr_labels.append((ax, t, entries))


def _obstacles(ax) -> list[Bbox]:
    """Display-space boxes for every marker and line sample in ``ax``."""
    px = ax.figure.dpi / 72
    boxes = []
    for line in ax.lines:
        xy = ax.transData.transform(np.column_stack(line.get_data()).astype(float))
        if len(xy) == 0:
            continue
        if line.get_marker() not in (None, "None", "", " "):
            r = (line.get_markersize() / 2 + line.get_markeredgewidth()) * px
            boxes += [Bbox.from_extents(x - r, y - r, x + r, y + r) for x, y in xy]
        if line.get_linestyle() not in ("None", "", " ") and len(xy) > 1:
            r = max(line.get_linewidth(), 1) * px
            for (x0, y0), (x1, y1) in zip(xy[:-1], xy[1:], strict=True):
                for f in np.linspace(0, 1, 24):
                    x, y = x0 + f * (x1 - x0), y0 + f * (y1 - y0)
                    boxes.append(Bbox.from_extents(x - r, y - r, x + r, y + r))
    return boxes


def _place_labels(fig) -> None:
    """Place each queued label at its first candidate that stays inside the plot and
    clears every marker, line and earlier label; drop it if none does (legend covers it)."""
    renderer = fig.canvas.get_renderer()
    for ax, t, entries in fig._jrr_labels:
        frame = ax.get_window_extent(renderer)
        taken = _obstacles(ax)
        pad = 1.5 * fig.dpi / 72
        for text, xy, candidates in entries:
            for dx, dy, ha, va in candidates:
                ann = ax.annotate(text, xy, xytext=(dx, dy), textcoords="offset points",
                                  ha=ha, va=va, fontsize=LABEL_PT, color=t["ink2"],
                                  annotation_clip=False)
                bb = ann.get_window_extent(renderer)
                inside = (bb.x0 >= frame.x0 and bb.x1 <= frame.x1 and bb.y0 >= frame.y0
                          and bb.y1 <= frame.y1)
                grown = Bbox.from_extents(bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad)
                if inside and not any(grown.overlaps(o) for o in taken):
                    taken.append(bb)
                    break
                ann.remove()


def _save(fig, out_dir: Path, name: str, theme: str) -> list[Path]:
    h = fig.get_size_inches()[1]
    fig.tight_layout(rect=(0, 0, 1, 1 - fig._jrr_header / h), w_pad=2.5)
    _place_labels(fig)
    paths = []
    for ext in ("png", "svg"):
        p = out_dir / f"{name}-{theme}.{ext}"
        fig.savefig(p, dpi=200, facecolor=fig.get_facecolor())
        paths.append(p)
    plt.close(fig)
    return paths


def _runs(summary: dict):
    for ds, block in summary["datasets"].items():
        if "-" not in ds:
            yield ds, block["runs"]


def _labelled(runs: dict, system: str, prefix: str) -> list[dict]:
    return [r for r in runs.values() if r["system"] == system and r["label"].startswith(prefix)
            and r["label"][len(prefix):].isdigit()]


# --------------------------------------------------------------------------- charts


def speed_accuracy(summary: dict, t: dict) -> plt.Figure:
    data = list(_runs(summary))
    fig, axes = _figure(t, len(data), "Speed vs accuracy",
                        "p50 latency per query (log scale) against nDCG@10 with 95% CI. "
                        "Up and to the left is better; the thin gray line is the Pareto "
                        "frontier.")
    seen: dict[str, Line2D] = {}
    for ax, (ds, runs) in zip(axes, data, strict=False):
        pts, labels, xs, ys = [], {}, [], []
        for s in MAIN:
            r = runs.get(s)
            if not r:
                continue
            x, n = r["latency"]["p50"] * 1000, r["metrics"]["ndcg@10"]
            pts.append((x, n["mean"]))
            xs.append(x)
            ys += n["ci"]
            ax.plot([x, x], n["ci"], color=_color(t, s), lw=1.5, solid_capstyle="butt",
                    zorder=2)
            ax.plot(x, n["mean"], MARKERS[s], ms=MARKER_PT, color=_color(t, s),
                    mec=t["surface"], mew=2, zorder=3)
            seen.setdefault(s, _marker_handle(t, s))
            labels[s] = (LABELS[s], (x, n["mean"]), AROUND)
        frontier, best = [], -1.0
        for x, y in sorted(pts):
            if y > best:
                frontier.append((x, y))
                best = y
        if len(frontier) > 1:
            ax.plot(*zip(*frontier, strict=True), color=t["muted"], lw=1, zorder=1,
                    solid_capstyle="round")
        _log_axis(ax, "x", xs, pad=2.2)
        if ys:
            span = max(ys) - min(ys) or 0.1
            ax.set_ylim(min(ys) - 0.12 * span, max(ys) + 0.18 * span)
        ax.yaxis.set_major_formatter(_fixed(2))
        ax.set_xlabel("p50 latency (ms)")
        ax.set_ylabel("nDCG@10")
        ax.set_title(DATASET_TITLES.get(ds, ds), loc="left", fontsize=11, pad=8,
                 color=t["ink"])
        _defer_labels(fig, ax, t, [labels[s] for s in LABEL_PRIORITY if s in labels])
    order = [s for s in MAIN if s in seen]
    _fig_legend(fig, t, [seen[s] for s in order], [LABELS[s] for s in order])
    return fig


def accuracy(summary: dict, t: dict) -> plt.Figure:
    data = list(_runs(summary))
    fig, axes = _figure(t, len(data), "Retrieval accuracy",
                        "nDCG@10 with 95% bootstrap CI. Mark vs cross-encoder: ▲ better, "
                        "= tie, ▼ worse (paired randomization test, p < 0.05).")
    seen: dict[str, Patch] = {}
    for ax, (ds, runs) in zip(axes, data, strict=False):
        present = [s for s in MAIN if s in runs]
        present.sort(key=lambda s: runs[s]["metrics"]["ndcg@10"]["mean"])
        ce = runs.get("dense+ce", {}).get("metrics", {}).get("ndcg@10", {}).get("mean")
        for y, s in enumerate(present):
            n = runs[s]["metrics"]["ndcg@10"]
            _bar(ax, y, n["mean"], _color(t, s), horizontal=True)
            seen.setdefault(s, Patch(facecolor=_color(t, s)))
            ax.plot(n["ci"], [y, y], color=t["ink2"], lw=1.2, solid_capstyle="butt", zorder=3)
            p = runs[s].get("p_vs_ce")
            mark = ""
            if p is not None and ce is not None:
                mark = "=" if p >= 0.05 else ("▲" if n["mean"] > ce else "▼")
            ax.annotate(f"{n['mean']:.3f} {mark}".strip(), (max(n["ci"][1], n["mean"]), y),
                        xytext=(5, 0), textcoords="offset points", va="center",
                        fontsize=LABEL_PT, color=t["ink2"])
        ax.set_yticks(range(len(present)), [LABELS[s] for s in present])
        ax.set_ylim(-0.6, len(present) - 0.4)
        ax.set_xlim(0, 1.0)
        ax.xaxis.set_major_formatter(_fixed(1))
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("nDCG@10")
        ax.set_title(DATASET_TITLES.get(ds, ds), loc="left", fontsize=11, pad=8,
                 color=t["ink"])
    # Emphasis form: two Jev hues plus one gray, so the legend names the three roles.
    roles = [(s, LABELS[s]) for s in ("jevragrank", "jevrank") if s in seen]
    if any(s not in ("jevragrank", "jevrank") for s in seen):
        roles.append(("base", "Baselines"))
    _fig_legend(fig, t, [Patch(facecolor=t[s] if s != "base" else t["base"]) for s, _ in roles],
                [label for _, label in roles])
    return fig


def scaling(summary: dict, t: dict) -> plt.Figure:
    runs = summary["datasets"].get("scifact", {}).get("runs", {})
    series = {s: sorted(_labelled(runs, s, "n"), key=lambda r: r["corpus_size"])
              for s in ("jevrank", "jevragrank", "dense")}
    series = {s: rs for s, rs in series.items() if rs}
    nq = sorted({r["n_queries"] for rs in series.values() for r in rs if "n_queries" in r})
    queries = f", {nq[0]} queries" if len(nq) == 1 else ""
    fig, (ax,) = _figure(t, 1, "Latency as the corpus grows",
                         f"SciFact subsets{queries}. p50 latency per query (log–log).")
    xs, ys, ends = [], [], []
    for s, rs in series.items():
        pts = [(r["corpus_size"], r["latency"]["p50"] * 1000) for r in rs]
        xs += [x for x, _ in pts]
        ys += [y for _, y in pts]
        ax.plot(*zip(*pts, strict=True), "-" + MARKERS[s], color=_color(t, s), lw=2, ms=MARKER_PT,
                mec=t["surface"], mew=2, solid_capstyle="round", solid_joinstyle="round")
        ends.append((f"{pts[-1][1]:,.0f} ms", pts[-1], LINE_END))
    if xs:
        ax.set_xscale("log")
        ax.set_xlim(min(xs) / 1.25, max(xs) * 2.2)  # room for the end labels
        ax.xaxis.set_major_locator(FixedLocator(sorted(set(xs))))
        ax.xaxis.set_major_formatter(FuncFormatter(_num))
        ax.xaxis.set_minor_locator(NullLocator())
        _log_axis(ax, "y", ys, pad=1.8)
    ax.set_xlabel("documents in corpus")
    ax.set_ylabel("p50 latency (ms)")
    _defer_labels(fig, ax, t, ends)
    _fig_legend(fig, t, [_marker_handle(t, s, line=True) for s in series],
                [LABELS[s] for s in series])
    return fig


def depth(summary: dict, t: dict) -> plt.Figure:
    runs = summary["datasets"].get("scifact", {}).get("runs", {})
    fig, axes = _figure(t, 2, "How many candidates to rerank",
                        "SciFact. Rerank depth K (dense candidates) against accuracy and "
                        "latency.")
    axes[0].set_title("Accuracy", loc="left", fontsize=11, pad=8,
                 color=t["ink"])
    axes[1].set_title("Latency", loc="left", fontsize=11, pad=8,
                 color=t["ink"])
    ks_all: set[int] = set()
    shown = []
    ends: tuple[list, list] = ([], [])
    for s in ("jevragrank", "dense+ce"):
        rs = _labelled(runs, s, "k") + ([runs[s]] if s in runs else [])
        pts = sorted(((r["options"].get("candidates", 50), r) for r in rs), key=lambda p: p[0])
        if not pts:
            continue
        shown.append(s)
        ks = [k for k, _ in pts]
        ks_all.update(ks)
        style = dict(color=_color(t, s), lw=2, ms=MARKER_PT, mec=t["surface"], mew=2,
                     solid_capstyle="round", solid_joinstyle="round")
        axes[0].plot(ks, [r["metrics"]["ndcg@10"]["mean"] for _, r in pts],
                     "-" + MARKERS[s], **style)
        lat = [r["latency"]["p50"] * 1000 for _, r in pts]
        axes[1].plot(ks, lat, "-" + MARKERS[s], **style)
        ends[0].append((LABELS[s], (ks[-1], pts[-1][1]["metrics"]["ndcg@10"]["mean"]), LINE_END))
        ends[1].append((LABELS[s], (ks[-1], lat[-1]), LINE_END))
    axes[0].set_ylabel("nDCG@10")
    axes[0].yaxis.set_major_formatter(_fixed(2))
    axes[1].set_ylabel("p50 latency (ms)")
    axes[1].yaxis.set_major_formatter(FuncFormatter(_num))
    for ax, entries in zip(axes, ends, strict=True):
        ax.set_xlabel("candidates reranked (K)")
        if ks_all:
            ax.set_xscale("log")
            ax.set_xlim(min(ks_all) / 1.25, max(ks_all) * 1.25)
            ax.xaxis.set_major_locator(FixedLocator(sorted(ks_all)))
            ax.xaxis.set_major_formatter(FuncFormatter(_num))
            ax.xaxis.set_minor_locator(NullLocator())
        ax.margins(y=0.15)
        _defer_labels(fig, ax, t, entries)
    axes[1].set_ylim(bottom=0)  # after the margins, so the top keeps its headroom
    _fig_legend(fig, t, [_marker_handle(t, s, line=True) for s in shown],
                [LABELS[s] for s in shown])
    return fig


def ablations(summary: dict, t: dict) -> plt.Figure:
    data = list(_runs(summary))
    fig, axes = _figure(t, 2, "Ablations",
                        "nDCG@10 averaged over datasets. Lighter column: the 0.8B decider.")
    axes[0].set_title("Jev question type (JevRAGRank)", loc="left", fontsize=11, pad=8,
                 color=t["ink"])
    axes[1].set_title("decider model size", loc="left", fontsize=11, pad=8,
                 color=t["ink"])
    strategies = [("score", ""), ("noul", "noul"), ("choice", "choice")]
    vals = []
    for name, label in strategies:
        key = "jevragrank" + (f"@{label}" if label else "")
        xs = [runs[key]["metrics"]["ndcg@10"]["mean"] for _, runs in data if key in runs]
        if xs:
            vals.append((name, sum(xs) / len(xs)))
    for i, (_name, v) in enumerate(vals):
        _bar(axes[0], i, v, t["jevragrank"], horizontal=False, thickness=BAR_MAX_PX)
        axes[0].annotate(f"{v:.3f}", (i, v), xytext=(0, 4), textcoords="offset points",
                         ha="center", va="bottom", fontsize=LABEL_PT, color=t["ink2"])
    axes[0].set_xticks(range(len(vals)), [n for n, _ in vals])
    axes[0].set_xlim(-0.6, max(len(vals), 1) - 0.4)
    axes[0].set_ylabel("nDCG@10")

    ax = axes[1]
    under = ax.get_xaxis_transform()  # x in data, y in axes fraction
    # Pair columns 10px apart (not the 2px touching gap) so their cap values never meet.
    half = BAR_MAX_PX / 2 + 5
    for i, s in enumerate(("jevragrank", "jevrank")):
        for side, (size, key) in ((-1, ("0.8B", f"{s}@0.8b")), (1, ("2B", s))):
            xs = [runs[key]["metrics"]["ndcg@10"]["mean"] for _, runs in data if key in runs]
            if not xs:
                continue
            v = sum(xs) / len(xs)
            color = _mix(t[s], t["surface"], 0.5) if size == "0.8B" else t[s]
            _bar(ax, i, v, color, horizontal=False, offset=side * half, thickness=BAR_MAX_PX)
            dx = side * half * 0.75  # px -> pt
            ax.annotate(f"{v:.3f}", (i, v), xytext=(dx, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=LABEL_PT, color=t["ink2"])
            ax.annotate(size, (i, 0), xycoords=under, xytext=(dx, -5),
                        textcoords="offset points", ha="center", va="top", fontsize=9,
                        color=t["ink2"])
        ax.annotate(LABELS[s], (i, 0), xycoords=under, xytext=(0, -20),
                    textcoords="offset points", ha="center", va="top", fontsize=9,
                    color=t["ink"])
    ax.set_xticks([0, 1], ["", ""])
    ax.set_xlim(-0.6, 1.6)
    for a in axes:
        a.set_ylim(0, 1)
        a.yaxis.set_major_formatter(_fixed(1))
        a.grid(axis="x", visible=False)
    return fig


CHARTS: dict[str, Callable[[dict, dict], plt.Figure]] = {
    "speed-accuracy": speed_accuracy, "accuracy": accuracy, "scaling": scaling,
    "depth": depth, "ablations": ablations}


def render_all(summary: dict, out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
                         "svg.fonttype": "none"})
    paths: list[Path] = []
    for theme, t in THEMES.items():
        for name, fn in CHARTS.items():
            paths += _save(fn(summary, t), out_dir, name, theme)
    return paths
