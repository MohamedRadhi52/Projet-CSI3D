# Architecture & Pipeline du Projet — Progressive Mesh (CSI 3D)

Ce document décrit l’organisation complète du dépôt, le rôle de chaque dossier et fichier, ainsi que le fonctionnement détaillé du pipeline de simplification progressive, d’évaluation géométrique et de streaming Web.

---

## 1. Vue d’ensemble du pipeline

```text
  Maillage .obj (example/suzanne.obj, example/bunny.obj)
         │
         ▼
  ┌──────────────────────────────────────────────────────────────────────┐
  │ 1. MOTEUR PYTHON : progressive_mesh.py (ProgressiveMesh)             │
  │    • Parsing & topologie (obja.Model + collect_edges)                │
  │    • Critère au choix :                                              │
  │        - Hoppe (1996)        ──► énergies dans progressive_mesh.py   │
  │        - QEM (Garland 1997)  ──► qem.py (quadriques 10 floats)       │
  │        - Saillance (2016)    ──► saillance.py (LCE) + qem.py         │
  │        - Chen (2002)         ──► chen.py (Half-Edge multi-niveaux)   │
  │    • Tas-min + garde-fous topologiques + collapses inversibles       │
  │    • Option --mode quantifie ──► quantification 16 bits              │
  └──────┬───────────────────────────────────────────────────────────────┘
         │  write_obja() : base M⁰ + vertex splits inversés + jalons "s"
         ▼
  Fichier streamable .obja (example/<mode>/<modele>0.1.obja)
         │
         ├───────────────────────────────────────┐
         ▼                                       ▼
  ┌─────────────────────────────────┐     ┌──────────────────────────────────┐
  │ 2. ÉVALUATION & COURBES         │     │ 3. VISUALISEUR WEB PROGRESSIF    │
  │  • comparaison.py               │     │  • server.py (HTTP Range GET)    │
  │  • generate_benchmark_plots.py  │     │  • index.html + js/main.js       │
  │  • render_slides.py             │     │  • js/obja.js (Loader + QoS +    │
  │  • generate_m0_chen.py          │     │    reconstruction Three.js)      │
  └─────────────────────────────────┘     └──────────────────────────────────┘
```

---

## 2. Emplacement et rôle des dossiers et fichiers

### À la racine du projet

| Fichier | Rôle |
|---|---|
| `progressive_mesh.py` | **Moteur principal et CLI** : contient `Collapse`, `ProgressiveMesh(obja.Model)`, `run_simplification()` et `main()`. Pilote les 5 modes (`hoppe`, `qem`, `quantifie`, `saillance`, `chen`), la quantification et l’export `.obja`. |
| `obja.py` | **Parseur et générateur OBJA** : définit `Face`, `Model` (lecture `.obj`/`.obja`) et `Output` (écriture des instructions `v`, `ev`, `f`, `ef`). |
| `qem.py` | **Module QEM (Garland & Heckbert 1997)** : initialise les quadriques $4\times 4$ symétriques sur 10 flottants (`compute_initial_quadrics`), ajoute les plans de contrainte de bord, choisit la position optimale (`optimal_candidate_position`) et met à jour $Q_t \leftarrow Q_t + Q_s$ (`update_quadrics_on_collapse`). |
| `saillance.py` | **Module Saillance (Limper et al. 2016)** : calcule la courbure moyenne discrète (`courbure_moyenne`), les distances géodésiques par Dijkstra (`voisinage_geodesique`) et l’entropie locale sur 5 rayons (`calculer_saillance`) pour pondérer les coûts QEM. |
| `chen.py` | **Module Half-Edge & préservation de forme (Chen & Nishita 2002)** : calcule les normales (`compute_face_normals`), verrouille les *sharp edges* et bords par palier (`classify_unremovable_vertices`) et choisit le sens de contraction `vs -> vt` (`choose_half_edge_direction`). |
| `comparaison.py` | **Évaluation géométrique** : calcule les distances plus proche voisin sommet-à-sommet (`_nearest_distances`), la distance de Hausdorff (`maximum_geometric_error`), l’erreur moyenne (Middlebury) et les taux de réduction (`compare_models`, `compare_files`). |
| `generate_benchmark_plots.py` | **Générateur des courbes** : rejoue chaque `.obja` aux jalons `s` et génère les 4 figures comparatives (Hausdorff, Middlebury, Complétude à $\tau = 2\%$ diag., Tailles) dans `example/renders/`. |
| `render_slides.py` | **Rendu qualitatif de $M^0$** : extrait le maillage de base $M^0$ vers `.obj` (`extract_base_mesh_obj`) et produit les grilles comparatives Matplotlib 3D (`Poly3DCollection`). |
| `generate_m0_chen.py` | **Rendu $M^0$ dédié à Chen & Nishita** : génère `example/renders/m0_chen.png` sous les mêmes angles que les autres méthodes. |
| `server.py` | **Serveur HTTP local avec support `Range`** : permet au navigateur de télécharger les fichiers `.obja` par morceaux d’octets (streaming progressif). |
| `index.html` | **Interface Web Three.js** : affiche la scène 3D, la barre de progression, la pause, la capture d’écran et les sélecteurs QoS (`QoS Adaptatif` / `Fixe`, profils réseau `Réseau réel` / `1 Mbps` / `64 kbps`). |
| `decimate.py` | Script TP d’origine minimaliste (`Decimater`, décimation naïve conservée comme référence de départ). |
| `Makefile` | Règle d’assemblage d’origine (`src/Loader.js` + `src/Model.js` $\rightarrow$ `js/obja.js`). |
| `README.md` | **Ce document** : architecture du projet, emplacement des fichiers et pipeline complet. |
| `Presentation_CSI3D.pdf` | Support de présentation (export PDF de la présentation Google Slides de soutenance). |

### Dossiers du projet

| Dossier | Contenu et rôle |
|---|---|
| `example/` | Modèles 3D de référence (`suzanne.obj`, `bunny.obj`) et quelques `.obja` de test à la racine du dossier. |
| `example/hoppe/` | Sorties `.obja` progressives générées avec le critère de Hoppe (`suzanne0.1.obja`, `bunny0.1.obja`). |
| `example/qem/` | Sorties `.obja` générées avec QEM (Garland & Heckbert). |
| `example/quantifie/` | Sorties `.obja` QEM quantifiées sur 16 bits. |
| `example/saillance/` | Sorties `.obja` QEM pondérées par la saillance LCE de Limper. |
| `example/chen/` | Sorties `.obja` générées avec Chen & Nishita (*Half-Edge Collapse*, 0 `ev`). |
| `example/renders/` | Images PNG finales des courbes (`courbe_1_hausdorff.png` à `courbe_4_tailles.png`), des rendus $M^0$ (`m0_*.png`), des grilles comparatives, ainsi que les `.obj` extraits pour $M^0$. |
| `js/` | Scripts exécutés par le navigateur : `three.min.js`, `OrbitControls.js`, `main.js` (contrôle de la scène et du streaming) et `obja.js` (client HTTP Range, contrôleur QoS adaptatif et reconstruction `Model.manageElement`). |
| `src/` | Sources JavaScript de base du TP (`Loader.js`, `Model.js`). *(Note : les évolutions QoS actuelles ont été intégrées directement dans `js/obja.js`.)* |
| `tests/` | Tests unitaires : `test_chen.py` (teste le pipeline final Chen : `ev_written == 0` et réversibilité complète par `vertex_split`) et `test_progressive_mesh.py` (anciens tests de la phase `.pmz` intermédiaire). |

---

## 3. Déroulement détaillé du pipeline

### Étape 1 — Chargement et tables d’incidence (`progressive_mesh.py`)
1. `ProgressiveMesh.parse_file()` (hérité de `obja.Model`) charge `vertices` (coordonnées 3D) et `faces` (triangles visibles).
2. `collect_edges()` parcourt les faces visibles une seule fois et construit trois dictionnaires d’incidence :
   - `edge_faces[(u, v)]` : indices des faces adjacentes à chaque arête (`len == 1` identifie les bords) ;
   - `vertex_faces[v]` : faces incidentes au sommet `v` ;
   - `vertex_neighbors[v]` : voisins directs (1-ring) de `v`.

### Étape 2 — Initialisation selon la méthode (`run_simplification` & `simplify`)
- **QEM (`--mode qem` / `quantifie` / `saillance`)** : `compute_initial_quadrics()` calcule pour chaque sommet sa quadrique sur 10 flottants (somme des plans des triangles pondérés par leur aire + plans perpendiculaires sur les arêtes de bord avec `boundary_weight=10.0`).
- **Saillance (`--mode saillance`)** : `saillance.calculer_saillance(self)` attribue un score dans $[0, 1]$ à chaque sommet (courbure discrète sur 256 classes + entropie de Shannon sur 5 rayons géodésiques). Ce score multiplie le coût QEM par $(1 + w \cdot \text{saillance})$.
- **Chen & Nishita (`--mode chen`)** : `chen.compute_face_normals()` calcule les normales des faces. La simplification s’exécute par paliers successifs `levels` : `((0.15, True), (0.40, True), (0.85, True), (1.40, False), (2.10, False))` où le flottant règle le seuil de courbure et le booléen (`lock_sharp_edges`) protège les arêtes vives (`normal_diff >= 0.85`) sur les trois premiers paliers avant de les débloquer si besoin pour atteindre `target_ratio`.
- **Hoppe (`--mode hoppe`)** : évalue l’énergie combinée `_distance_energy` + `_spring_energy` + `_discontinuity_energy` + terme de représentation.

### Étape 3 — File de priorité, validation et contraction (`simplify`)
1. `_build_priority_queue()` remplit le tas-min `self._heap` avec les arêtes candidates.
2. À chaque tour, tant que `count_visible_faces() > target_faces` :
   - `shortest_edge()` dépile l’arête de plus faible coût en ignorant les entrées obsolètes via `self._edge_versions` (*lazy deletion*).
   - `is_valid_collapse(vs, vt, candidate)` vérifie 4 garde-fous : pas de contraction d’arête de bord isolée (`len == 1`), pas de fusion bord/intérieur (`_is_boundary_vertex`), respect de la *Link Condition* (`_link_condition`, préserve le 2-manifold) et non-retournement des normales autour du 1-ring.
   - `collapse_edge(vs, vt)` fusionne `vs` vers `vt` placé en `candidate` (en mode Chen, `candidate = vertices[vt]`, donc `vt` reste fixe), met à jour **localement** les faces et les tables d’incidence (`edge_faces`, `vertex_faces`, `vertex_neighbors`), réinsère les arêtes voisines modifiées dans `_heap` et enregistre un objet `Collapse` dans `self.history`.

### Étape 4 — Quantification géométrique 16 bits (`quantize_model`)
En mode `quantifie`, `quantize_model(bits=16)` calcule la boîte englobante globale de toutes les positions (base $M^0$ + tous les splits de `history`), projette chaque axe sur une grille entière $[0, 65\,535]$, puis déquantifie aussitôt vers des flottants arrondis (`dequantize_vertices`). Le fichier `.obja` écrit donc des coordonnées flottantes compatibles avec Three.js, mais alignées sur la grille 16 bits.

### Étape 5 — Export progressif `.obja` (`write_obja`)
1. Écrit d’abord le **maillage de base $M^0$** (sommets et faces restés visibles à 10 %).
2. Parcourt `reversed(self.history)` pour traduire chaque `Collapse` en **vertex split** (`v` pour recréer `vs`, `ev` si `vt` avait bougé — sauté automatiquement en mode Chen —, `ef` pour rebrancher les faces modifiées et `f` pour réactiver les 2 faces supprimées).
3. Insère un marqueur de taille binaire estimée `s <octets>` après $M^0$ puis après chaque *split* (`vertex_size = (3 * bits + 7) // 8`, `face_size = 7`, `edit_face_size = 5`).

### Étape 6 — Évaluation et streaming Web
- **Évaluation (`generate_benchmark_plots.py` + `comparaison.py`)** : rejoue le `.obja` aux jalons `s` et mesure Hausdorff, erreur moyenne (Middlebury) et complétude géométrique ($\tau = 2\%$ de la diagonale) par rapport au `.obj` original.
- **Visualiseur Web (`server.py` + `index.html` + `js/obja.js`)** : `Loader` télécharge le `.obja` par requêtes HTTP `Range`, ajuste la taille des blocs selon la bande passante mesurée ou simulée (QoS), et `Model.manageElement()` met à jour la géométrie Three.js en temps réel.

---

## 4. Commandes utiles

```bash
# Générer les 5 modes sur Suzanne à 10 %
python progressive_mesh.py example/suzanne.obj --ratio 0.1 --mode all

# Générer un seul mode (ex. Chen & Nishita ou QEM quantifié 16 bits)
python progressive_mesh.py example/bunny.obj --ratio 0.1 --mode chen
python progressive_mesh.py example/bunny.obj --ratio 0.1 --mode quantifie --bits 16

# Générer les 4 graphiques comparatifs et le rendu M^0 Chen
python generate_benchmark_plots.py
python generate_m0_chen.py

# Lancer le serveur Web puis ouvrir http://localhost:8000/?example/qem/bunny0.1.obja
python server.py

# Lancer le test unitaire Chen
python -m unittest tests/test_chen.py
```
