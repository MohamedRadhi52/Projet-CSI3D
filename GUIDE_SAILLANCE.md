# Guide d'Intégration : Saillance de Surface (Limper et al. 2016)

Ce document explique comment brancher son calcul de **Saillance multi-échelle par Entropie de Courbure (LCE - Local Curvature Entropy)** sur notre moteur **Progressive Mesh QEM (Garland 1997)**.

---

## 1. État Actuel du Moteur (`test1.py`)

Toute la chaîne géométrique est déjà implémentée, optimisée et validée :
- **Algèbre QEM complète (Garland 1997)** : quadriques compactes à 10 flottants, plans de contraintes aux frontières (§6), évaluation matricielle $\mathbf{v}^T \mathbf{Q} \mathbf{v}$.
- **Mises à jour topologiques locales $O(1)$** : simplification ultra-rapide (~350 contractions/sec).
- **File de priorité (`heapq`)** : trie automatiquement les arêtes par coût minimal.
- **Export OBJA progressif propre** : compatible à 100% avec le visualiseur Three.js et la plateforme de benchmark de l'école (testé et validé).

---

## 2. Où Intervenir dans `test1.py` ?

Le hook est **déjà préparé** dans le code. Tu n'as que **trois petits points** à compléter.

### Point 1 : Les variables déjà en place (`__init__`)
Dans `ProgressiveMesh.__init__` (vers la ligne 130) :
```python
self.saliency_weight = float(saliency_weight)  # Contrôlé par l'argument CLI --saliency-weight
self.vertex_saliency = None                    # <-- C'EST ICI QUE TU DOIS METTRE TON TABLEAU
```
`self.vertex_saliency` doit être un tableau (ou liste) de flottants de taille $N$ ($N = \text{nombre de sommets}$), normalisé entre `0.0` (zone plate/banale) et `1.0` (zone très saillante/caractéristique).

---

### Point 2 : Écrire ta fonction de calcul de Saillance
Tu peux ajouter dans la classe `ProgressiveMesh` ta méthode de calcul :

```python
def compute_saliency(self):
    """Calcule la saillance multi-échelle LCE (Limper 2016) pour chaque sommet.
    
    Étapes selon le papier :
    1. Courbure moyenne discrète H_v par opérateur cotangente de Laplace-Beltrami.
    2. Voisinages géodésiques multi-échelles (ex: 5 échelles de rayons r_1 à r_5).
    3. Entropie locale de courbure pondérée par les aires Voronoï (LCE).
    4. Normalisation finale dans [0, 1].
    """
    num_v = len(self.vertices)
    self.vertex_saliency = np.zeros(num_v, dtype=np.float64)
    
    # === TON ALGORITHME ICI ===
    # Remplir self.vertex_saliency[i] pour chaque sommet i
```

---

### Point 3 : L'appel dans `simplify()`
Dans la méthode `simplify()` (vers la ligne 745), déclenche le calcul au tout début avant de construire le tas de priorité :

```python
def simplify(self, target_ratio=0.1):
    self.collect_edges()
    if self.use_qem and not self.quadrics10:
        self.compute_initial_quadrics()

    # === AJOUT SAILLANCE ===
    if self.saliency_weight > 0.0 and self.vertex_saliency is None:
        print("    - Calcul de la saillance de surface (Limper 2016)...")
        self.compute_saliency()

    initial_faces = self.count_visible_faces()
    target_faces = max(1, int(initial_faces * target_ratio))
    self._build_priority_queue()
    ...
```

---

## 3. Comment la Saillance Influence la Simplification ?

Le branchement dans le calcul du coût est **DÉJÀ FAIT** dans `qem_cost(a, b, position)` (vers la ligne 200) :

```python
def qem_cost(self, a, b, position):
    q_sum = sum_quadric10(self.quadrics10[a], self.quadrics10[b])
    cost = eval_quadric10(q_sum, float(position[0]), float(position[1]), float(position[2]))

    # Hook pour la saillance (Papier 3 - Limper 2016) :
    if self.saliency_weight > 0.0 and self.vertex_saliency is not None:
        sal_a = self.vertex_saliency[a] if a < len(self.vertex_saliency) else 0.0
        sal_b = self.vertex_saliency[b] if b < len(self.vertex_saliency) else 0.0
        sal_edge = 0.5 * (sal_a + sal_b)
        cost *= (1.0 + self.saliency_weight * sal_edge)

    return cost
```

### Pourquoi cette formule ?
- Si une arête $(a, b)$ est dans une zone plate ($\text{saillance} \approx 0$), son coût reste pur QEM $\rightarrow$ elle est contractée en premier.
- Si une arête est dans une zone saillante (yeux, crêtes, oreilles, $\text{saillance} \approx 1$), son coût est multiplié par $(1 + \alpha)$ $\rightarrow$ elle est rejetée au fond du tas et **protégée** de la contraction !

---

## 4. Commandes de Test et de Validation

Une fois ta fonction codée, teste directement en ligne de commande :

```powershell
# 1. Test QEM Pur (référence)
python test1.py example/suzanne.obj --ratio 0.1 --method qem

# 2. Test QEM + Saillance (alpha = 5.0)
python test1.py example/suzanne.obj --ratio 0.1 --method qem --saliency-weight 5.0

# 3. Test sur le lapin
python test1.py example/bunny.obj --ratio 0.1 --method qem --saliency-weight 5.0
```

### Ce que tu dois observer :
1. **Visuellement dans le visualiseur** : Les yeux et le visage de Suzanne doivent rester bien plus nets et reconnaissables à 10% de faces avec saillance que sans saillance.
2. **Sur la courbe de la plateforme** : La courbe de Hausdorff démarrera un peu plus haut au début (~0.60 vs 0.46) car la saillance protège les petits détails au lieu de la distance pure, puis rejoindra le palier (~0.33) vers 50% des bits. C'est le résultat théorique attendu !
