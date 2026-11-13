#!/usr/bin/env python3
"""Extrait les maillages de base M^0 (10%) pour Chen & Nishita (2002)
et génère le visuel côte à côte example/renders/m0_chen.png.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image, ImageChops

import comparaison
from render_slides import extract_base_mesh_obj

RENDERS_DIR = ROOT / "example" / "renders"
RENDERS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    suzanne_obj = RENDERS_DIR / "suzanne_chen_10.obj"
    bunny_obj = RENDERS_DIR / "bunny_chen_10.obj"
    extract_base_mesh_obj(str(ROOT / "example" / "chen" / "suzanne0.1.obja"), suzanne_obj)
    extract_base_mesh_obj(str(ROOT / "example" / "chen" / "bunny0.1.obja"), bunny_obj)

    fig = plt.figure(figsize=(11.0, 4.0), dpi=200)
    fig.patch.set_facecolor("white")

    specs = [
        ("Suzanne M^0 — 10% (96 faces)", suzanne_obj, 78, -90, False),
        ("Bunny M^0 — 10% (496 faces)", bunny_obj, 18, -45, True),
    ]
    for i, (title, obj_path, elev, azim, is_bunny) in enumerate(specs, 1):
        ax = fig.add_subplot(1, 2, i, projection="3d")
        ax.set_facecolor("white")
        m = comparaison._load_mesh(obj_path)
        verts = np.array([v for v in m.vertices])
        faces = np.array([[f.a, f.b, f.c] for f in m.faces if getattr(f, "visible", True)])
        v_plot = np.column_stack([verts[:, 0], verts[:, 2], verts[:, 1]]) if is_bunny else verts
        tris = v_plot[faces]
        v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
        normals = np.cross(v1 - v0, v2 - v0)
        norm = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.where(norm > 0, normals / norm, 0)
        l1 = np.array([0.4, 0.4, 0.8])
        l1 /= np.linalg.norm(l1)
        l2 = np.array([-0.5, 0.3, 0.6])
        l2 /= np.linalg.norm(l2)
        shade = np.clip(
            0.45 + 0.40 * np.maximum(0, normals @ l1) + 0.20 * np.maximum(0, normals @ l2),
            0.45,
            1.0,
        )
        rgb = np.array([0xC0 / 255.0, 0x39 / 255.0, 0x2B / 255.0])
        poly = Poly3DCollection(
            tris,
            facecolors=shade[:, None] * rgb,
            edgecolors="#1e272e",
            linewidths=0.4 if is_bunny else 0.55,
            alpha=1.0,
        )
        ax.add_collection3d(poly)
        c = (v_plot.max(axis=0) + v_plot.min(axis=0)) * 0.5
        r = 1.08 if not is_bunny else 0.075
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)
        ax.view_init(elev=elev, azim=azim)
        ax.axis("off")
        ax.set_title(title, color="#1e272e", fontsize=14, pad=2, fontweight="bold")

    plt.subplots_adjust(left=0.01, right=0.99, top=0.92, bottom=0.01, wspace=0.02)
    out = RENDERS_DIR / "m0_chen.png"
    plt.savefig(out, facecolor="white", edgecolor="none", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)

    im = Image.open(out).convert("RGB")
    bg = Image.new("RGB", im.size, (255, 255, 255))
    bbox = ImageChops.difference(im, bg).getbbox()
    if bbox:
        pad = 16
        crop_box = (
            max(0, bbox[0] - pad),
            max(0, bbox[1] - pad),
            min(im.width, bbox[2] + pad),
            min(im.height, bbox[3] + pad),
        )
        im = im.crop(crop_box)
        im.save(out)
    print(f"Saved {out}: {im.size}")


if __name__ == "__main__":
    main()
