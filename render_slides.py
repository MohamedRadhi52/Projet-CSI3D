#!/usr/bin/env python3
"""
Script utilitaire pour extraire les maillages .obj et générer les images comparatives
qualitatives pour la soutenance (Google Slides).
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import comparaison

RENDERS_DIR = Path("example/renders")
RENDERS_DIR.mkdir(parents=True, exist_ok=True)

def extract_base_mesh_obj(obja_path, out_obj_path):
    """Extrait le maillage de base M^0 (10% de faces) d'un fichier .obja vers un .obj standard."""
    with open(obja_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    base_lines = []
    seen_face = False
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        tag = parts[0]
        if tag == "v" and not seen_face:
            base_lines.append(line)
        elif tag == "f":
            seen_face = True
            base_lines.append(line)
        elif seen_face and tag in ["v", "ev", "efv", "df"]:
            break
            
    with open(out_obj_path, "w", encoding="utf-8") as f:
        f.writelines(base_lines)
    print(f"-> Extrait {out_obj_path} ({len(base_lines)} lignes)")

def render_comparison_grid(models, out_path, elev=78, azim=-90, is_bunny=False):
    """Génère une vue comparative 4 panneaux avec éclairage studio et wireframe discret."""
    fig = plt.figure(figsize=(19, 5.5), dpi=250)
    fig.patch.set_facecolor("#ffffff")
    
    for i, (title, path, hex_col) in enumerate(models):
        ax = fig.add_subplot(1, 4, i + 1, projection="3d")
        ax.set_facecolor("#ffffff")
        
        m = comparaison._load_mesh(Path(path))
        verts = np.array([v for v in m.vertices])
        faces = np.array([[f.a, f.b, f.c] for f in m.faces if getattr(f, "visible", True)])
        
        if is_bunny:
            v_plot = np.column_stack([verts[:, 0], verts[:, 2], verts[:, 1]])
        else:
            v_plot = verts
            
        mesh_tris = v_plot[faces]
        
        v0 = mesh_tris[:, 0]
        v1 = mesh_tris[:, 1]
        v2 = mesh_tris[:, 2]
        normals = np.cross(v1 - v0, v2 - v0)
        norm = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.where(norm > 0, normals / norm, 0)
        
        l1 = np.array([0.4, 0.4, 0.8]); l1 /= np.linalg.norm(l1)
        l2 = np.array([-0.5, 0.3, 0.6]); l2 /= np.linalg.norm(l2)
        diff1 = np.maximum(0, np.dot(normals, l1))
        diff2 = np.maximum(0, np.dot(normals, l2))
        shade = np.clip(0.45 + 0.40 * diff1 + 0.20 * diff2, 0.45, 1.0)
        
        h = hex_col.lstrip("#")
        rgb = np.array([int(h[j:j+2], 16)/255.0 for j in (0, 2, 4)])
        face_colors = shade[:, None] * rgb
        
        poly = Poly3DCollection(mesh_tris, facecolors=face_colors, edgecolors="#1e272e", linewidths=0.4 if is_bunny else 0.55, alpha=1.0)
        ax.add_collection3d(poly)
        
        c = (v_plot.max(axis=0) + v_plot.min(axis=0)) * 0.5
        r = 1.35 if not is_bunny else 0.095
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)
        
        ax.view_init(elev=elev, azim=azim)
        ax.axis("off")
        ax.set_title(title, color="#1e272e", fontsize=13, pad=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, facecolor=fig.get_facecolor(), edgecolor="none", bbox_inches="tight")
    print(f"-> Rendu sauvegardé : {out_path}")

def main():
    print("=== 1. Extraction des maillages de base M^0 (.obj) ===")
    extract_base_mesh_obj("example/qem/suzanne0.1.obja", RENDERS_DIR / "suzanne_qem_10.obj")
    extract_base_mesh_obj("example/saillance/suzanneW0.1.obja", RENDERS_DIR / "suzanne_saillance_10.obj")
    extract_base_mesh_obj("example/hoppe/suzanne0.1.obja", RENDERS_DIR / "suzanne_hoppe_10.obj")
    
    extract_base_mesh_obj("example/qem/bunny0.1.obja", RENDERS_DIR / "bunny_qem_10.obj")
    extract_base_mesh_obj("example/saillance/bunnyR0.1.obja", RENDERS_DIR / "bunny_saillance_10.obj")
    extract_base_mesh_obj("example/hoppe/bunny0.1.obja", RENDERS_DIR / "bunny_hoppe_10.obj")

    print("\n=== 2. Génération des grilles comparatives HD ===")
    suzanne_models = [
        ("Original (968 faces)", "example/suzanne.obj", "#27ae60"),
        ("Hoppe 10% (96 faces)", RENDERS_DIR / "suzanne_hoppe_10.obj", "#8e44ad"),
        ("QEM 10% (96 faces)", RENDERS_DIR / "suzanne_qem_10.obj", "#2980b9"),
        ("QEM + Saillance 10% (96 faces)", RENDERS_DIR / "suzanne_saillance_10.obj", "#d35400"),
    ]
    render_comparison_grid(suzanne_models, RENDERS_DIR / "comparaison_suzanne_10percent.png", elev=78, azim=-90, is_bunny=False)

    bunny_models = [
        ("Original (4 968 faces)", "example/bunny.obj", "#27ae60"),
        ("Hoppe 10% (496 faces)", RENDERS_DIR / "bunny_hoppe_10.obj", "#8e44ad"),
        ("QEM 10% (496 faces)", RENDERS_DIR / "bunny_qem_10.obj", "#2980b9"),
        ("QEM + Saillance 10% (496 faces)", RENDERS_DIR / "bunny_saillance_10.obj", "#d35400"),
    ]
    render_comparison_grid(bunny_models, RENDERS_DIR / "comparaison_bunny_10percent.png", elev=18, azim=-45, is_bunny=True)
    print("\nTerminé avec succès !")

if __name__ == "__main__":
    main()
