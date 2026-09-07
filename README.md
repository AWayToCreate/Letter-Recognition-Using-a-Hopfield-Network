# 🧠 Reconnaissance de lettres avec un réseau de Hopfield

Projet Python implémentant un réseau de Hopfield capable de
reconnaître des lettres à partir d'images bruitées.

## 🎯 Objectif

L'objectif de ce projet est d'étudier le fonctionnement d'un
réseau de Hopfield et sa capacité à reconstruire un motif
à partir d'une version contenant du bruit.

## 🛠️ Technologies

- Python
- NumPy
- Pillow
- Matplotlib

## ⚙️ Fonctionnement

Le programme :

1. Charge des images de lettres au format PPM
2. Transforme les images en matrices binaires
3. Construit la matrice de poids du réseau
4. Ajoute du bruit aux lettres
5. Reconstruit les motifs grâce au réseau
6. Calcule la similarité
7. Identifie la lettre reconnue

## 📊 Expérimentation

Le programme teste différents niveaux de bruit :

- 10 %
- 30 %
- 45 %

Les résultats permettent d'observer l'évolution de
l'énergie du réseau et de la similarité avec le motif original.

## 🚀 Installation

Installer les dépendances :

```bash
pip install -r requirements.txt
