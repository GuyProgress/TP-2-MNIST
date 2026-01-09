# TP2 & TP5 - Visualisation et Réduction de Dimension sur MNIST et jumeaux hybrides

## 📋 Table des matières
- [Description](#description)
- [TP2 : Réduction de Dimension](#tp2--réduction-de-dimension)
- [TP2 - P2 : Auto-encodeurs](#tp5--auto-encodeurs)
- [Installation](#installation)
- [Structure du Projet](#structure-du-projet)
- [Utilisation](#utilisation)
- [Méthodes Implémentées](#méthodes-implémentées)
- [Résultats](#résultats)

## 📝 Description

Ce projet explore différentes techniques de réduction de dimension et de visualisation appliquées à la base de données de chiffres manuscrits (digits dataset de scikit-learn). Il combine des méthodes classiques de réduction de dimension (TP2) avec des approches d'apprentissage profond utilisant des auto-encodeurs (TP5).

**Dataset utilisé :** Digits (sklearn)
- 1797 échantillons
- Images 8x8 pixels (64 features)
- 10 classes (chiffres 0-9)

## 🎯 TP2 : Réduction de Dimension

### Objectifs
- Explorer et visualiser des données de haute dimension
- Comparer différentes méthodes de réduction de dimension
- Analyser la séparabilité des classes dans l'espace réduit

### Méthodes Implémentées

#### 1. **PCA (Principal Component Analysis)**
- Méthode linéaire basée sur la variance
- Rapide et déterministe
- Préserve la variance globale des données
- Analyse de la variance expliquée cumulative

#### 2. **t-SNE (t-Distributed Stochastic Neighbor Embedding)**
- Méthode non-linéaire
- Excellente pour la visualisation
- Préserve les structures locales
- Tests avec différentes valeurs de perplexité (5, 30, 50)
- Optimisation de l'exagération initiale

#### 3. **MDS (Multidimensional Scaling)**
- Préserve les distances entre points
- Maintient la structure géométrique
- Comparaison métrique vs non-métrique
- Analyse du stress (qualité de la représentation)

#### 4. **UMAP (Uniform Manifold Approximation and Projection)**
- Équilibre entre structure locale et globale
- Plus rapide que t-SNE
- Bonne préservation de la topologie

## 🤖 TP5 : Auto-encodeurs

### Objectifs
- Apprendre des représentations compressées par apprentissage profond
- Reconstruire les données à partir de l'espace latent
- Comparer avec les méthodes classiques

### Architecture Implémentée

#### Auto-encodeur 2D
```
Encodeur: 64 → 32 (ReLU) → 16 (ReLU) → 2 (Linear)
Décodeur: 2 → 16 (ReLU) → 32 (ReLU) → 64 (Sigmoid)
```

#### Auto-encodeur 3D
```
Encodeur: 64 → 32 (ReLU) → 16 (ReLU) → 3 (Linear)
Décodeur: 3 → 16 (ReLU) → 32 (ReLU) → 64 (Sigmoid)
```

### Caractéristiques
- Fonction de perte : MSE (Mean Squared Error)
- Optimiseur : Adam
- 100 époques d'entraînement
- Batch size : 32
- Validation split : 20%

### Visualisations
- Évolution de la perte pendant l'entraînement
- Espace latent 2D et 3D
- Reconstruction des images originales
- Comparaison qualitative des reconstructions

## 🛠️ Installation

### Prérequis
```bash
Python 3.8+
```

### Dépendances
```bash
pip install numpy
pip install matplotlib
pip install pandas
pip install seaborn
pip install scikit-learn
pip install tensorflow  # Pour les auto-encodeurs
pip install umap-learn  # Optionnel pour UMAP
```

### Installation rapide
```bash
pip install -r requirements.txt
```

## 📁 Structure du Projet

```
TP 2 MNIST/
├── TP2_MNIST.ipynb          # Notebook principal
├── Readme.md                # Ce fichier
└── requirements.txt         # Dépendances Python
```

## 🚀 Utilisation

### Lancer le notebook
1. Ouvrir Jupyter Notebook ou VS Code
2. Charger `TP2_MNIST.ipynb`
3. Exécuter les cellules séquentiellement

### Sections du notebook
1. **Chargement des données** - Import et exploration du dataset
2. **Visualisation initiale** - Affichage d'exemples de chiffres
3. **PCA** - Réduction linéaire et analyse de variance
4. **t-SNE** - Exploration avec différents paramètres
5. **MDS** - Préservation des distances
6. **UMAP** - Méthode moderne (optionnel)
7. **Auto-encodeurs** - Apprentissage profond
8. **Comparaison** - Vue d'ensemble de toutes les méthodes

## 📊 Méthodes Implémentées

| Méthode | Type | Avantages | Inconvénients |
|---------|------|-----------|---------------|
| **PCA** | Linéaire | Rapide, déterministe, interprétable | Ne capture pas les relations non-linéaires |
| **t-SNE** | Non-linéaire | Excellente séparation visuelle | Lent, non-déterministe, perd structure globale |
| **MDS** | Distance | Préserve distances, structure géométrique | Coûteux en calcul, sensible au bruit |
| **UMAP** | Non-linéaire | Rapide, équilibre local/global | Nécessite installation supplémentaire |
| **Auto-encodeur** | Deep Learning | Reconstruction, flexible, non-linéaire | Nécessite entraînement, hyperparamètres |

## 📈 Résultats

### Variance Expliquée (PCA)
- 2 composantes : ~25% de variance
- 95% variance nécessite ~21 composantes

### Séparation des Classes
- **t-SNE** : Meilleure séparation visuelle des clusters
- **MDS** : Bonne préservation de la structure géométrique
- **Auto-encodeur** : Séparation comparable, avec capacité de reconstruction

### Reconstruction (Auto-encodeur)
- MSE finale : ~0.01-0.02
- Visualisation fidèle des chiffres après reconstruction

## 🔍 Analyses Complémentaires

### Optimisation t-SNE
- Perplexité optimale : 30-40
- Early exaggeration : 12-20
- Impact significatif sur la qualité visuelle

### MDS Métrique vs Non-métrique
- Métrique : Préserve distances exactes
- Non-métrique : Plus flexible, préserve l'ordre

### Auto-encodeur 2D vs 3D
- 2D : Visualisation directe
- 3D : Meilleure capacité de représentation

## 📚 Références

- **PCA**: Pearson, K. (1901). "On Lines and Planes of Closest Fit to Systems of Points in Space"
- **t-SNE**: van der Maaten & Hinton (2008). "Visualizing Data using t-SNE"
- **MDS**: Kruskal, J.B. (1964). "Multidimensional scaling by optimizing goodness of fit"
- **UMAP**: McInnes et al. (2018). "UMAP: Uniform Manifold Approximation and Projection"
- **Auto-encodeurs**: Hinton & Salakhutdinov (2006). "Reducing the Dimensionality of Data with Neural Networks"

## 👥 Auteur

Projet réalisé dans le cadre des TPs de visualisation de données et apprentissage automatique.

## 📄 Licence

Ce projet est à usage éducatif.

---

**Note**: Pour de meilleures performances, il est recommandé d'exécuter le notebook sur une machine avec au moins 8GB de RAM. Les auto-encodeurs bénéficient d'un GPU mais peuvent fonctionner sur CPU.
