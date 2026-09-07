import random       # pour générer des nombres aléatoires (choix de neurones, bruit)
import os           # pour manipuler les chemins de fichiers
import sys          # pour quitter le programme si nécessaire
from PIL import Image  # pour ouvrir et traiter les images PPM
import numpy as np      # pour manipuler facilement les tableaux d'images
import matplotlib.pyplot as plt  # pour tracer l'évolution de l'énergie et de la similarité

# --- Fonctions de base ---

def signe(x):
    """Renvoie 1 si x >= 0, sinon -1."""
    return 1 if x >= 0 else -1

def ppm_to_matrix(filepath, threshold=128, size=(15, 15)):
    """Convertit une image PPM en matrice binaire (-1 / 1).
    
    Variables :
    - filepath : chemin vers le fichier image
    - threshold : seuil pour distinguer les pixels sombres et clairs
    - size : taille finale de la matrice (lignes, colonnes)
    """
    img = Image.open(filepath).convert("L")  # conversion en niveaux de gris
    img = img.resize(size)                   # redimensionnement à taille souhaitée
    mat = np.array(img)                      # conversion en tableau NumPy
    mat = np.where(mat < threshold, 1, -1)  # pixels < seuil -> 1, sinon -1
    return mat.tolist()                      # retour en liste de listes

def charger_motifs_ppm(dossier, lettres_cibles=("A", "B", "C")):
    """Charge les lettres depuis un dossier en format PPM.
    
    Variables :
    - dossier : dossier contenant les fichiers .ppm
    - lettres_cibles : lettres à charger
    - motifs : dictionnaire {lettre: matrice binaire}
    """
    motifs = {}
    for lettre in lettres_cibles:
        chemin = os.path.join(dossier, f"{lettre}.ppm")
        if os.path.exists(chemin):
            motifs[lettre] = ppm_to_matrix(chemin)
        else:
            print(f"[⚠️] Image manquante : {lettre}.ppm")
    return motifs

def afficher(matrice):
    """Affiche une matrice avec des blocs ASCII (█ pour 1, espace pour -1).
    
    Variables :
    - matrice : liste de listes contenant 1/-1
    """
    for ligne in matrice:
        ligne_affichee = ""
        for x in ligne:
            if x == 1:
                ligne_affichee += "█"
            else:
                ligne_affichee += " "
        print(ligne_affichee)
    print()

def aplatir(matrice):
    """Transforme une matrice 2D en vecteur 1D.
    
    Variables :
    - matrice : liste de listes
    - resultat : liste 1D contenant tous les éléments
    """
    resultat = []
    for ligne in matrice:
        for x in ligne:
            resultat.append(x)
    return resultat

def en_matrice(vecteur, taille=(15, 15)):
    """Transforme un vecteur 1D en matrice 2D.
    
    Variables :
    - vecteur : liste 1D
    - taille : tuple (lignes, colonnes)
    - matrice : liste de listes résultat
    """
    lignes, colonnes = taille
    matrice = []
    for i in range(lignes):
        debut = i * colonnes
        fin = (i + 1) * colonnes
        ligne = vecteur[debut:fin]
        matrice.append(ligne)
    return matrice

def creation_T(motifs):
    """Crée la matrice de poids du réseau de Hopfield.
    
    Variables :
    - motifs : dictionnaire de motifs appris {lettre: matrice binaire}
    - N : nombre de neurones (taille du vecteur aplati)
    - T : matrice N x N des poids synaptiques
    - v : vecteur aplati d'un motif
    """
    # Déterminer le nombre de neurones à partir du premier motif
    for premier_motif in motifs.values():
        break
    N = len(aplatir(premier_motif))
    
    # Initialisation de la matrice T à zéro
    T = []
    for i in range(N):
        ligne = []
        for j in range(N):
            ligne.append(0)
        T.append(ligne)
    
    # Règle d'apprentissage de Hebb
    for motif in motifs.values():
        v = aplatir(motif)
        for i in range(N):
            for j in range(N):
                if i != j:  # pas d'auto-connexion
                    T[i][j] += v[i] * v[j]
    return T

def bruiter(motif, proportion=0.1):
    """Inverse aléatoirement un pourcentage de pixels.
    
    Variables :
    - motif : matrice originale
    - proportion : fraction de bits à inverser
    - v : motif aplati
    - n : nombre de pixels à bruiter
    """
    v = aplatir(motif)
    n = int(len(v) * proportion)
    for _ in range(n):
        i = random.randint(0, len(v)-1)
        v[i] *= -1
    return v

def energie(v, T):
    """Calcule l'énergie du réseau pour un état donné.
    
    Variables :
    - v : vecteur état
    - T : matrice de poids
    - E : énergie totale
    """
    E = 0
    N = len(v)
    for i in range(N):
        for j in range(N):
            E += T[i][j] * v[i] * v[j]
    return -0.5 * E

def similarite(v, ref):
    """Renvoie le pourcentage de bits identiques entre v et ref.
    
    Variables :
    - v : vecteur courant
    - ref : vecteur référence
    - identiques : compteur de bits identiques
    """
    n = len(v)
    identiques = 0
    for i in range(n):
        if v[i] == ref[i]:
            identiques += 1
    return identiques / n * 100

def rappel(v, T, motifs, iterations=1000, trace=False, nom_lettre="?"):
    """Rappel asynchrone avec suivi d'énergie et similarité.
    
    Variables :
    - v : vecteur initial (potentiellement bruité)
    - T : matrice de poids
    - motifs : dictionnaire des motifs appris
    - iterations : nombre d'itérations de rappel
    - trace : si True, affichage de l'évolution
    - nom_lettre : lettre cible pour calcul de similarité
    - energies : liste des énergies enregistrées
    - similitudes : liste des similarités enregistrées
    """
    energies = []
    similitudes = []

    for t in range(iterations):
        # Choisir un neurone aléatoire
        i = random.randint(0, len(v) - 1)
        
        # Calcul de la somme pondérée pour le neurone i
        somme = 0
        for j in range(len(v)):
            somme += T[i][j] * v[j]
        
        # Mise à jour du neurone
        v[i] = signe(somme)
        
        # Enregistrement de l'énergie et similarité toutes les 10 itérations
        if trace and t % 10 == 0:
            E = energie(v, T)
            energies.append(E)
            ref = aplatir(motifs[nom_lettre])
            sim = similarite(v, ref)
            similitudes.append(sim)
            print("Itération", t, "- Énergie :", E, "- Similarité :", sim, "%")
    
    return v, energies, similitudes

def reconnaitre(v, motifs):
    """Compare l'état final avec chaque motif appris et retourne le plus proche.
    
    Variables :
    - v : vecteur final du réseau
    - motifs : dictionnaire {lettre: motif}
    - distances : dictionnaire {lettre: nombre de différences}
    """
    distances = {}
    for nom, motif in motifs.items():
        ref = aplatir(motif)
        diff = 0
        for i in range(len(v)):
            if v[i] != ref[i]:
                diff += 1
        distances[nom] = diff
    
    # Trouver le motif avec la distance minimale
    plus_proche = None
    min_diff = None
    for nom in distances:
        if min_diff is None or distances[nom] < min_diff:
            min_diff = distances[nom]
            plus_proche = nom
    return plus_proche

# --- Programme principal ---
if __name__ == "__main__":
    dossier = "lettres_ppm"
    motifs = charger_motifs_ppm(dossier, lettres_cibles=("A", "B", "C"))

    if not motifs:
        print("[ERREUR] Aucun motif trouvé dans le dossier lettres_ppm/")
        sys.exit()

    T = creation_T(motifs)  # matrice de poids du réseau

    mot_original = "ABC"
    niveaux_bruit = [0.1, 0.3, 0.45]  # différents niveaux de bruit
    mot_reconstruit = ""

    print("\n--- MOTS ORIGINAUX ---")
    for lettre in mot_original:
        print(f"\nLettre {lettre} :")
        afficher(motifs[lettre])

    # Historique pour enregistrer courbes énergie/similarité
    historique = {}

    for bruit_niveau in niveaux_bruit:
        historique[bruit_niveau] = {}
        print(f"\n\n=== TEST avec bruit {int(bruit_niveau*100)}% ===")
        for lettre in mot_original:
            if lettre not in motifs:
                continue

            # Génération du motif bruité
            bruit = bruiter(motifs[lettre], proportion=bruit_niveau)
            print(f"\nLettre {lettre} bruitée ({int(bruit_niveau*100)}%) :")
            afficher(en_matrice(bruit))

            # Rappel du motif avec affichage progressif
            reconstruit, energies, similitudes = rappel(
                bruit, T, motifs, iterations=100, trace=True, nom_lettre=lettre
            )

            # Reconnaissance finale
            lettre_reconnue = reconnaitre(reconstruit, motifs)
            mot_reconstruit += lettre_reconnue
            historique[bruit_niveau][lettre] = (energies, similitudes)

            print(f"\nLettre reconnue : {lettre_reconnue}")
            afficher(en_matrice(reconstruit))

    # --- Graphiques ---
    fig, ax1 = plt.subplots(figsize=(10,6))

    couleurs_lettres = {"A": "tab:blue", "B": "tab:green", "C": "tab:red"}
    styles_bruit = {0.1: "-", 0.3: "--", 0.45: "-."}

    # Axe énergie
    for bruit_niveau, lettres_data in historique.items():
        for lettre, (energies, similitudes) in lettres_data.items():
            if not energies:
                continue
            x = range(0, len(energies)*10, 10)
            couleur = couleurs_lettres.get(lettre, "black")
            style = styles_bruit.get(bruit_niveau, "-")
            ax1.plot(x, energies, color=couleur, linestyle=style,
                     label=f"{lettre} - {int(bruit_niveau*100)}% énergie")

    ax1.set_xlabel("Itérations")
    ax1.set_ylabel("Énergie", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    # Axe similarité
    ax2 = ax1.twinx()
    for bruit_niveau, lettres_data in historique.items():
        for lettre, (energies, similitudes) in lettres_data.items():
            if not similitudes:
                continue
            x = range(0, len(similitudes)*10, 10)
            couleur = couleurs_lettres.get(lettre, "black")
            style = styles_bruit.get(bruit_niveau, "--")
            ax2.plot(x, similitudes, color=couleur, linestyle=style, alpha=0.7,
                     label=f"{lettre} - {int(bruit_niveau*100)}% similarité")

    ax2.set_ylabel("Similarité (%)", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    # Légendes combinées
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="lower right", fontsize=8)

    plt.title("Évolution de l'énergie (↙) et de la similarité (↗) par lettre et bruit")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
