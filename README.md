# 🧠 Reconnaissance de lettres avec un réseau de Hopfield

Projet Python implémentant un **réseau de Hopfield** capable de reconnaître et de reconstruire des lettres à partir d'images contenant différents niveaux de bruit.

## 🎯 Objectif

L'objectif de ce projet est d'étudier le fonctionnement d'un réseau de Hopfield et sa capacité à retrouver un motif mémorisé à partir d'une version bruitée.

Le réseau apprend trois motifs correspondant aux lettres **A, B et C**, puis tente de les reconstruire lorsqu'une partie de leurs pixels est modifiée.

## 🛠️ Technologies

- 🐍 Python
- 🔢 NumPy
- 🖼️ Pillow
- 📊 Matplotlib

## ⚙️ Fonctionnement

Le programme suit plusieurs étapes :

1. Charge les images des lettres au format `.ppm`
2. Redimensionne les images en matrices de `15 × 15`
3. Transforme les pixels en valeurs binaires `-1 / 1`
4. Construit la matrice de poids du réseau
5. Applique la règle d'apprentissage de Hebb
6. Ajoute différents niveaux de bruit aux lettres
7. Reconstruit les motifs grâce au réseau de Hopfield
8. Calcule l'énergie du réseau
9. Mesure la similarité avec le motif original
10. Identifie la lettre reconnue

## 🧠 Réseau de Hopfield

Chaque pixel de l'image est représenté par un neurone.

Avec des images de `15 × 15`, le réseau utilise donc **225 neurones**.

Les lettres A, B et C sont utilisées comme motifs mémorisés par le réseau. Lorsqu'un motif bruité est présenté au réseau, les neurones sont mis à jour progressivement afin de retrouver un état correspondant à l'un des motifs appris.

## 📊 Expérimentation

Le programme teste trois niveaux de bruit :

- **10 %**
- **30 %**
- **45 %**

Pour chaque niveau de bruit, le programme observe :

- l'évolution de l'énergie du réseau ;
- l'évolution de la similarité avec le motif original ;
- la reconstruction de la lettre ;
- la lettre finalement reconnue.

Les résultats sont représentés graphiquement avec **Matplotlib**.

## 📁 Structure du projet

```text
reseau_de_neurones/
│
├── lettres_ppm/
│   ├── A.ppm
│   ├── B.ppm
│   └── C.ppm
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
