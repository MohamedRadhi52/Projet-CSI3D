test1.py est le programme de transformation:

1. Chargement du modèle
2. Recherche des arêtes: collect_edges()
3. Calcul du coût : edge_length(a,b) à améliorer
4. Choix de l'arête à contracter: shortest_edge()  à Améliorer
5. Edge Collapse : collapse_edge(vs, vt)
6. Suppression des faces dégénérées : collapse_edge()
7. Sauvegarde de l'historique: collapse_edge() à améliorer
8. Répéter : simplify()
9. Critère d'arrêt : simplify(target_ratio=0.1). Très basique (10% de la taille d’origine)
10. Construction du maillage de base : write_obja()
11. Reconstruction progressive : write_obja() à Améliorer grâce à la sauvegarde de l’historique


Ce qu’il manque: 
Pas de vraie structure Vertex Split
Calcul du coût basique
Pas de voisinage
Pas d'optimisation mémoire
Pas de gestion topologique avancée

