#!/usr/bin/env python3
"""Génère les 4 graphiques comparatifs (Hausdorff, Middlebury, Complétude, Tailles)
à partir des fichiers .obja produits pour les 5 options (Hoppe, QEM, Quantifié 16b, Saillance, Chen & Nishita).
"""

import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import comparaison
import obja

RENDERS_DIR = ROOT / "example" / "renders"
RENDERS_DIR.mkdir(parents=True, exist_ok=True)

MODES = [
    ("hoppe", "Hoppe (1996)", "#8E44AD", "-", 2.2, 1.0),
    ("qem", "QEM (Garland 1997)", "#2980B9", "-", 2.4, 0.95),
    ("quantifie", "QEM + Quantifié (16b)", "#16A085", "--", 2.0, 1.0),
    ("saillance", "QEM + Saillance (Limper 2016)", "#D35400", "-", 2.1, 0.95),
    ("chen", "Chen & Nishita (2002)", "#C0392B", "-", 2.1, 0.95),
]


def evaluate_progressive_obja(orig_path, obja_path, num_points=55):
    """Rejoue le flux .obja ligne par ligne et calcule les métriques sur num_points paliers s."""
    orig = comparaison._load_mesh(str(orig_path))
    orig_verts = comparaison._vertices_as_array(orig)
    diag = float(np.linalg.norm(orig_verts.max(axis=0) - orig_verts.min(axis=0)))
    tau = 0.02 * diag

    lines = pathlib.Path(obja_path).read_text(encoding="utf-8").splitlines()
    raw_steps = []
    for line in lines:
        parts = line.split()
        if parts and parts[0] == "s" and len(parts) >= 2:
            raw_steps.append(int(parts[1]))

    total_checkpoints = len(raw_steps)
    sample_indices = set(
        np.linspace(0, total_checkpoints - 1, min(num_points, total_checkpoints), dtype=int).tolist()
    )

    cur_model = obja.Model()
    cp_idx = 0
    snapshots = []
    for line in lines:
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue
        tag = parts[0]
        if tag == "v" and len(parts) >= 4:
            cur_model.vertices.append(np.array([float(parts[1]), float(parts[2]), float(parts[3])]))
        elif tag == "f" and len(parts) >= 4:
            cur_model.faces.append(obja.Face(int(parts[1]) - 1, int(parts[2]) - 1, int(parts[3]) - 1))
        elif tag == "ev" and len(parts) >= 5:
            vi = int(parts[1]) - 1
            cur_model.vertices[vi] = np.array([float(parts[2]), float(parts[3]), float(parts[4])])
        elif tag == "ef" and len(parts) >= 5:
            fi = int(parts[1]) - 1
            cur_model.faces[fi] = obja.Face(int(parts[2]) - 1, int(parts[3]) - 1, int(parts[4]) - 1)
        elif tag == "s" and len(parts) >= 2:
            s_val = int(parts[1])
            if cp_idx in sample_indices:
                vis_v = set()
                for f in cur_model.faces:
                    if f.visible:
                        vis_v.update([f.a, f.b, f.c])
                pts = np.asarray([cur_model.vertices[i] for i in sorted(vis_v)], dtype=float)
                snapshots.append((s_val, pts))
            cp_idx += 1

    total_s = snapshots[-1][0]
    fracs, haus_list, mean_list, comp_list = [], [], [], []
    for s_val, pts in snapshots:
        d_o2f = comparaison._nearest_distances(orig_verts, pts)
        d_f2o = comparaison._nearest_distances(pts, orig_verts)
        fracs.append(s_val / total_s)
        haus_list.append(max(float(d_o2f.max()), float(d_f2o.max())))
        mean_list.append(float(np.mean(np.concatenate((d_o2f, d_f2o)))))
        comp_list.append(float(np.mean(d_o2f <= tau)))

    return {
        "frac": np.array(fracs),
        "hausdorff": np.array(haus_list),
        "middlebury": np.array(mean_list),
        "completeness": np.array(comp_list),
        "real_kb": os.path.getsize(obja_path) / 1024.0,
        "est_kb": total_s / 1024.0,
    }


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#CBD5E1",
        "axes.linewidth": 0.9,
        "grid.color": "#E2E8F0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.7,
    })

    data = {}
    for mesh in ("suzanne", "bunny"):
        orig_p = ROOT / "example" / f"{mesh}.obj"
        data[mesh] = {}
        for mode_key, *_ in MODES:
            obja_p = ROOT / "example" / mode_key / f"{mesh}0.1.obja"
            data[mesh][mode_key] = evaluate_progressive_obja(orig_p, obja_p, num_points=55)
            d = data[mesh][mode_key]
            print(
                f"{mesh}/{mode_key}: pts={len(d['frac'])}, H0={d['hausdorff'][0]:.4f}, "
                f"M0={d['middlebury'][0]:.4f}, C0={d['completeness'][0]:.2f}, "
                f"real={d['real_kb']:.1f}Ko, est={d['est_kb']:.1f}Ko"
            )

    def plot_two_panel(metric_key, title, ylabel, out_name, ylim=None):
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.7), dpi=220)
        fig.patch.set_facecolor("#FFFFFF")

        panel_titles = [
            ("suzanne", "Suzanne (968 faces → base 10%)"),
            ("bunny", "Bunny (4 968 faces → base 10%)"),
        ]
        for ax, (mesh_key, ptitle) in zip(axes, panel_titles):
            ax.set_facecolor("#F8FAFC")
            ax.grid(True, alpha=0.9)
            for mode_key, label, color, ls, lw, alpha in MODES:
                d = data[mesh_key][mode_key]
                ax.plot(
                    d["frac"],
                    d[metric_key],
                    label=label,
                    color=color,
                    linestyle=ls,
                    linewidth=lw,
                    alpha=alpha,
                    marker="o",
                    markersize=3.2,
                    markevery=4,
                )
            ax.set_title(ptitle, fontsize=11.5, fontweight="bold", color="#142341", pad=9)
            ax.set_xlabel("Fraction des données transmises (s / S_total)", fontsize=9.5, color="#334155")
            ax.set_ylabel(ylabel, fontsize=9.5, color="#334155")
            ax.set_xlim(0.0, 1.02)
            if ylim is not None:
                ax.set_ylim(*ylim)
            else:
                ymax = ax.get_ylim()[1]
                ax.set_ylim(bottom=0.0, top=ymax * 1.05)
            ax.tick_params(colors="#475569", labelsize=8.5)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.suptitle(title, fontsize=13.5, fontweight="bold", color="#142341", y=0.98)
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.91),
            ncol=3,
            frameon=True,
            facecolor="#FFFFFF",
            edgecolor="#CBD5E1",
            fontsize=8.9,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.80])
        out_path = RENDERS_DIR / out_name
        fig.savefig(out_path, facecolor="#FFFFFF", dpi=220, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")

    plot_two_panel(
        "hausdorff",
        "Distance de Hausdorff progressive — Comparaison des 5 options",
        "Distance de Hausdorff (plus bas = meilleur)",
        "courbe_1_hausdorff.png",
    )
    plot_two_panel(
        "middlebury",
        "Erreur géométrique moyenne (type Middlebury) — Comparaison des 5 options",
        "Erreur moyenne bidirectionnelle (plus bas = meilleur)",
        "courbe_2_middlebury.png",
    )
    plot_two_panel(
        "completeness",
        "Complétude progressive du maillage (seuil τ = 2 % de la diagonale)",
        "Complétude (fraction sommets ≤ τ)",
        "courbe_3_completude.png",
        ylim=(0.0, 1.04),
    )

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.7), dpi=220)
    fig.patch.set_facecolor("#FFFFFF")
    short_labels = ["Hoppe\n(32b)", "QEM\n(32b)", "Quantifié\n(16b)", "Saillance\n(32b)", "Chen\n(32b)"]
    mode_keys = ["hoppe", "qem", "quantifie", "saillance", "chen"]
    x = np.arange(len(mode_keys))
    w = 0.36

    for ax, (mesh_key, ptitle) in zip(axes, [
        ("suzanne", "Suzanne (base 10% → 100%)"),
        ("bunny", "Bunny (base 10% → 100%)"),
    ]):
        ax.set_facecolor("#F8FAFC")
        ax.grid(True, axis="y", alpha=0.9)
        real_vals = [data[mesh_key][m]["real_kb"] for m in mode_keys]
        est_vals = [data[mesh_key][m]["est_kb"] for m in mode_keys]

        b1 = ax.bar(
            x - w / 2,
            real_vals,
            width=w,
            label="Taille fichier .obja (texte ASCII)",
            color="#142341",
            edgecolor="#0F172A",
        )
        b2_colors = ["#2DB5BE", "#2DB5BE", "#16A085", "#2DB5BE", "#C0392B"]
        b2 = ax.bar(
            x + w / 2,
            est_vals,
            width=w,
            label="Taille binaire estimée (32b ; vert = 16b quantifié ; rouge = Chen 32b)",
            color=b2_colors,
            edgecolor="#0F172A",
        )

        for rect in b1:
            h = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                h + max(real_vals) * 0.015,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#142341",
                fontweight="bold",
            )
        for idx_b, rect in enumerate(b2):
            h = rect.get_height()
            lbl = f"{h:.1f}\n(-15%)" if idx_b == 2 else f"{h:.1f}"
            col_lbl = "#0F766E" if idx_b == 2 else "#0F172A"
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                h + max(real_vals) * 0.015,
                lbl,
                ha="center",
                va="bottom",
                fontsize=8,
                color=col_lbl,
                fontweight="bold",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, fontsize=9, color="#1E293B")
        ax.set_ylabel("Taille en Ko (kilo-octets)", fontsize=9.5, color="#334155")
        ax.set_ylim(0, max(real_vals) * 1.16)
        ax.set_title(ptitle, fontsize=11.5, fontweight="bold", color="#142341", pad=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Comparaison des tailles : fichier .obja (ASCII) vs binaire estimée", fontsize=13.5, fontweight="bold", color="#142341", y=0.98)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=2,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CBD5E1",
        fontsize=9.5,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.84])
    out4 = RENDERS_DIR / "courbe_4_tailles.png"
    fig.savefig(out4, facecolor="#FFFFFF", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out4}")


if __name__ == "__main__":
    main()
