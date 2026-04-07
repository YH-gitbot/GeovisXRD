# src/geovisxrd/plotting/causal_plots.py
"""
Causal graph visualisation — compact flowchart / logic-diagram style.

  - plot_causal_graph_networkx
"""

import os
import importlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx

from .._logging import get_logger
logger = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Node sizing — circular, text-driven
# ─────────────────────────────────────────────────────────────────────────────
_FONT_SIZE = 9
_CHAR_W    = 0.006   # data units per character
_CHAR_H    = 0.022   # data units per text line
_PAD_R     = 0.013   # extra radius around text

_LABEL_W   = 0.08
_LABEL_H   = 0.030


def _node_radius(label: str) -> float:
    return max(len(label) * _CHAR_W / 2, _CHAR_H / 2) + _PAD_R


def _circle_boundary(cx: float, cy: float, r: float,
                     tx: float, ty: float):
    """Return the point on circle (cx, cy, r) on the ray toward (tx, ty).

    Used to clip edge endpoints to the exact circle perimeter so that
    arcs start and end at the node boundary, not at the centre.
    """
    dx, dy = tx - cx, ty - cy
    dist = np.hypot(dx, dy)
    if dist < 1e-9:
        return cx + r, cy
    return cx + r * dx / dist, cy + r * dy / dist


# ─────────────────────────────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────────────────────────────

def _compute_layout(G):
    """graphviz neato/dot → kamada_kawai → spring."""
    n = max(G.number_of_nodes(), 1)
    for mod_path in ("networkx.drawing.nx_agraph", "networkx.drawing.nx_pydot"):
        try:
            mod = importlib.import_module(mod_path)
        except ImportError:
            continue
        for prog, args in [("neato", "-Goverlap=false -Gsplines=line"),
                           ("dot",   "")]:
            try:
                pos = (mod.graphviz_layout(G, prog=prog, args=args) if args
                       else mod.graphviz_layout(G, prog=prog))
                logger.info(f"Layout: graphviz {prog}")
                return pos
            except Exception:
                pass

    try:
        scale = max(3.0, n * 0.9)
        pos = nx.kamada_kawai_layout(G, scale=scale)
        logger.info("Layout: kamada_kawai")
        return pos
    except Exception:
        pass

    k_val = max(3.0, 6.0 / max(n ** 0.5, 1))
    pos = nx.spring_layout(G, seed=42, k=k_val, iterations=300, scale=3.0)
    logger.info(f"Layout: spring (k={k_val:.2f})")
    return pos


def _normalise_and_separate(pos, min_dist, max_iter=300):
    """Normalise to [0, 1] then push nodes apart until min_dist is satisfied."""
    nodes = list(pos.keys())
    n = len(nodes)
    if n == 0:
        return {}
    if n == 1:
        return {nodes[0]: (0.5, 0.5)}

    arr = np.array([pos[nd] for nd in nodes], dtype=float)
    lo, hi = arr.min(axis=0), arr.max(axis=0)
    span = np.where(hi - lo < 1e-6, 1.0, hi - lo)
    arr = (arr - lo) / span

    rng = np.random.default_rng(42)
    for _ in range(max_iter):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                delta = arr[i] - arr[j]
                dist  = float(np.linalg.norm(delta))
                if dist < min_dist:
                    if dist < 1e-9:
                        delta = rng.standard_normal(2)
                        dist  = float(np.linalg.norm(delta))
                    push    = (min_dist - dist) * 0.5 * delta / dist
                    arr[i] += push
                    arr[j] -= push
                    moved = True
        if not moved:
            break

    ctr = (arr.min(axis=0) + arr.max(axis=0)) / 2
    arr = arr - ctr + 0.5
    return {nd: (float(arr[i, 0]), float(arr[i, 1])) for i, nd in enumerate(nodes)}


# ─────────────────────────────────────────────────────────────────────────────
# Arc-radius assignment — prevent shared routes
# ─────────────────────────────────────────────────────────────────────────────

def _assign_arc_radii(filtered_edges, pos, bidir_keys,
                      base=0.15, dist_tol=0.30, angle_tol=25):
    """Assign per-edge arc3 curvature radii to minimise long shared routes.

    Strategy
    --------
    1. Bidirectional pairs (u→v AND v→u) get +base / −base (opposite curves).
    2. For all other edges, compute each edge's chord midpoint and direction.
       Edges whose chords are nearly parallel (angle difference < angle_tol)
       AND whose midpoints are close (< dist_tol) are considered overlapping.
    3. Connected components of the overlap graph are assigned spread-out radii
       so each edge in the group travels a clearly distinct arc.
    """
    n = len(filtered_edges)
    if n == 0:
        return []

    # Chord midpoint and normalised direction angle [0°, 180°) per edge
    chords = []
    for u, v, _ in filtered_edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        chords.append((
            ((x1 + x2) / 2, (y1 + y2) / 2),
            np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180,
        ))

    # Build pairwise overlap adjacency (skip bidirectional pairs)
    adj = [[] for _ in range(n)]
    for i in range(n):
        if (min(filtered_edges[i][0], filtered_edges[i][1]),
                max(filtered_edges[i][0], filtered_edges[i][1])) in bidir_keys:
            continue
        mid_i, ang_i = chords[i]
        for j in range(i + 1, n):
            if (min(filtered_edges[j][0], filtered_edges[j][1]),
                    max(filtered_edges[j][0], filtered_edges[j][1])) in bidir_keys:
                continue
            mid_j, ang_j = chords[j]
            if np.hypot(mid_j[0] - mid_i[0], mid_j[1] - mid_i[1]) > dist_tol:
                continue
            adiff = abs(ang_i - ang_j)
            adiff = min(adiff, 180 - adiff)  # → [0°, 90°]
            if adiff < angle_tol:
                adj[i].append(j)
                adj[j].append(i)

    # BFS: find connected components
    visited    = [False] * n
    comp_pos   = [0] * n   # position of edge within its component
    comp_size  = [1] * n

    for start in range(n):
        if visited[start]:
            continue
        comp, queue = [], [start]
        while queue:
            v = queue.pop(0)
            if visited[v]:
                continue
            visited[v] = True
            comp.append(v)
            queue.extend(nb for nb in adj[v] if not visited[nb])
        for k, idx in enumerate(comp):
            comp_pos[idx]  = k
            comp_size[idx] = len(comp)

    # Assign radii
    step = 0.08
    radii = []
    for i, (u, v, _) in enumerate(filtered_edges):
        key = (min(u, v), max(u, v))
        if key in bidir_keys:
            # Opposite curves for each direction
            rad = base if str(u) <= str(v) else -base
        elif comp_size[i] == 1:
            rad = base
        else:
            ng = comp_size[i]
            k  = comp_pos[i]
            # Spread radii symmetrically around base
            # For large groups, alternate sign to keep curves away from centre
            if ng <= 4:
                opts = [base + (k - (ng - 1) / 2) * step for k in range(ng)]
            else:
                # Alternate positive / negative with increasing magnitude
                half = (ng + 1) // 2
                pos_r = [base + j * step for j in range(half)]
                neg_r = [-(base + j * step) for j in range(ng - half)]
                opts  = []
                for a, b in zip(pos_r, neg_r + [None]):
                    opts.append(a)
                    if b is not None:
                        opts.append(b)
                if len(opts) < ng:
                    opts.append(pos_r[-1] + step)
            rad = opts[k % len(opts)]
        radii.append(rad)

    return radii


# ─────────────────────────────────────────────────────────────────────────────
# Edge label position
# ─────────────────────────────────────────────────────────────────────────────

def _arc_label_pos(p_src, p_tgt, rad, offset=0.028):
    """Approximate position of an edge label at the arc midpoint.

    The arc midpoint is offset from the chord midpoint perpendicular to the
    chord by approximately ``rad * chord_length / 2``.  An extra ``offset``
    nudges the label away from the arc line itself.
    """
    mx, my = (p_src[0] + p_tgt[0]) / 2, (p_src[1] + p_tgt[1]) / 2
    dx, dy = p_tgt[0] - p_src[0], p_tgt[1] - p_src[1]
    chord  = np.hypot(dx, dy)
    if chord < 1e-9:
        return mx + offset, my
    # Perpendicular unit vector (CCW rotation)
    px, py = -dy / chord, dx / chord
    total_offset = rad * chord / 2 + offset
    return mx + total_offset * px, my + total_offset * py


# ─────────────────────────────────────────────────────────────────────────────
# Label collision avoidance
# ─────────────────────────────────────────────────────────────────────────────

def _free_label_pos(lx, ly, occupied):
    s = 0.05
    candidates = [
        (0, 0),
        (0, s), (s, 0), (0, -s), (-s, 0),
        (s, s), (-s, s), (s, -s), (-s, -s),
        (0, 2*s), (2*s, 0), (0, -2*s), (-2*s, 0),
        (2*s, s), (-2*s, s),
    ]
    for ddx, ddy in candidates:
        cx, cy = lx + ddx, ly + ddy
        if not any(
            abs(cx - ox) < (_LABEL_W + ow) / 2 and
            abs(cy - oy) < (_LABEL_H + oh) / 2
            for ox, oy, ow, oh in occupied
        ):
            return cx, cy
    return lx, ly + 3 * s


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def plot_causal_graph_networkx(nx_graph, threshold=0.0, title="Causal Graph",
                               save_path=None, show=False):
    """Plot a causal graph with curved edges and compact circular nodes.

    Design
    ------
    Nodes   : small circles sized to wrap their label text.
    Edges   : arc3 curved edges drawn via ax.annotate.  Per-edge curvature
              radii are assigned to prevent multiple edges from sharing the
              same route for long distances.
    Bidir   : opposite-sign radii (+base / −base) so both directions curve
              away from each other.
    Overlap : edges with nearly parallel chords and close midpoints are spread
              across a range of radii (base ± k*step) so their arcs diverge.
    Arrows  : constant mutation_scale=13; shrinkA/B computed from actual node
              radii so arrowheads land exactly at the circle boundary.
    Labels  : placed at approximate arc midpoint; collision-aware fallback
              avoids overlap with nodes and other labels.
    Geometry: constant linewidth=1.5 and arrowhead size for all edges.
              Weight represented by colour (blue pos / red neg) + label only.

    Returns list of (u, v, weight) that passed the threshold filter.
    """
    logger.info(f"Plotting causal graph (threshold={threshold}).")

    # ── Filter ────────────────────────────────────────────────────────────────
    filtered_edges = []
    for u, v, d in nx_graph.edges(data=True):
        w = d.get("weight", 0)
        if abs(w) >= threshold:
            filtered_edges.append((u, v, w))

    filtered_graph = nx.DiGraph()
    filtered_graph.add_nodes_from(nx_graph.nodes())
    filtered_graph.add_weighted_edges_from(filtered_edges)

    n = filtered_graph.number_of_nodes()

    # ── Uniform node radius — sized to the longest label ─────────────────────
    # All nodes share one radius so the graph looks visually consistent and
    # no label can overflow its circle.
    r_uniform  = max((_node_radius(str(nd)) for nd in filtered_graph.nodes()),
                     default=0.025)
    node_radii = {nd: r_uniform for nd in filtered_graph.nodes()}
    min_dist   = 2 * r_uniform + 0.16

    # ── Layout + separation ───────────────────────────────────────────────────
    raw_pos = _compute_layout(filtered_graph)
    pos     = _normalise_and_separate(raw_pos, min_dist=min_dist)

    # ── Figure ────────────────────────────────────────────────────────────────
    fig_w = max(12, n * 2.0)
    fig_h = max(9,  n * 1.6)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(-0.20, 1.20)
    ax.set_ylim(-0.20, 1.20)

    # ── Bidirectional pairs & arc radii ───────────────────────────────────────
    edge_set   = {(u, v) for u, v, _ in filtered_edges}
    bidir_keys = {(min(u, v), max(u, v)) for u, v in edge_set if (v, u) in edge_set}

    arc_radii  = _assign_arc_radii(filtered_edges, pos, bidir_keys)

    # ── Occupied regions — seed with node circles ─────────────────────────────
    occupied = [
        (px, py, 2 * r + 0.02, 2 * r + 0.02)
        for nd, r in node_radii.items()
        for px, py in [pos[nd]]
    ]

    # ── Draw edges ────────────────────────────────────────────────────────────
    for i, (u, v, w) in enumerate(filtered_edges):
        color = "#1f77b4" if w >= 0 else "#d62728"
        rad   = arc_radii[i]

        # Clip start/end to the exact circle boundary using the straight-line
        # direction between centres.  shrinkA=0 / shrinkB=0 so FancyArrowPatch
        # does NOT apply any additional shortening — the arc already begins and
        # ends precisely on the node perimeter.
        bp_src = _circle_boundary(*pos[u], node_radii[u], *pos[v])
        bp_tgt = _circle_boundary(*pos[v], node_radii[v], *pos[u])

        ax.annotate(
            "", xy=bp_tgt, xytext=bp_src,
            arrowprops=dict(
                arrowstyle="-|>",
                connectionstyle=f"arc3,rad={rad:.3f}",
                color=color,
                lw=1.5,
                mutation_scale=13,
                shrinkA=0,
                shrinkB=0,
            ),
            zorder=2, annotation_clip=False,
        )

        # Label at arc midpoint, collision-aware
        lx, ly = _arc_label_pos(pos[u], pos[v], rad)
        lx, ly = _free_label_pos(lx, ly, occupied)
        occupied.append((lx, ly, _LABEL_W, _LABEL_H))

        ax.text(
            lx, ly, f"{w:+.2f}",
            fontsize=7.5, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.12",
                      facecolor="white", edgecolor="none", alpha=0.80),
            zorder=3,
        )

    # ── Draw nodes (on top of edges) ──────────────────────────────────────────
    for nd in filtered_graph.nodes():
        px, py = pos[nd]
        r = node_radii[nd]
        ax.add_patch(mpatches.Circle(
            (px, py), r,
            linewidth=1.0,
            edgecolor="steelblue",
            facecolor="white",
            zorder=4,
        ))
        ax.text(
            px, py, str(nd),
            fontsize=_FONT_SIZE, fontweight="bold",
            ha="center", va="center",
            zorder=5,
        )

    ax.set_title(f"{title} (threshold={threshold})", fontsize=12, pad=12)

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Causal graph saved to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return filtered_edges
