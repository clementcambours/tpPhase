#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 16:39:34 2025

@author: snahmani
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy import stats

# Pour chercher des chaînes de caractère dans des fichiers
import re


def grep_file(pattern, filename):
    """Recherche un motif dans un fichier et retourne les lignes correspondantes."""
    results = []
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            if re.search(pattern, line):
                results.append(line.strip())  # Supprime les espaces et les sauts de ligne

    return results  # Retourne les lignes trouvées


def plot_residual_analysis(A, B, dP_est, figure_title=None, save_path=None):
    """
    Calcule les résidus (V_est = B - A @ dP_est) et trace une figure contenant :
      1. La série temporelle des résidus (affichée en points)
      2. L'histogramme des résidus (nombre d'observations par bin)
      3. Le Q-Q Plot des résidus
      4. Un scatter plot des résidus en fonction des valeurs prédites
      5. Un encart affichant quelques statistiques (moyenne, variance, écart-type, skewness, kurtosis)
    
    Paramètres :
      - A : array-like, matrice des variables explicatives
      - B : array-like, vecteur des observations
      - dP_est : array-like, vecteur des paramètres estimés
      - figure_title (optionnel) : str, titre global de la figure
      - save_path (optionnel) : str, chemin complet (avec nom et extension) pour sauvegarder la figure
      
    Renvoie :
      - fig : l'objet Figure de matplotlib contenant l'ensemble des graphiques
    """
    
    # Calcul des résidus et des valeurs prédites
    V_est = B - A @ dP_est
    B_est = A @ dP_est
    
    # Calcul des statistiques sur les résidus
    moyenne    = np.mean(V_est)
    variance   = np.var(V_est)
    ecart_type = np.std(V_est)
    skewness   = stats.skew(V_est)
    kurtosis   = stats.kurtosis(V_est)
    
    # Création d'une figure avec GridSpec : 3 lignes et 2 colonnes
    # La dernière ligne (row 2) est utilisée pour l'encart des statistiques et s'étend sur les 2 colonnes.
    fig = plt.figure(figsize=(15, 15))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.5])
    
    # 1. Série temporelle des résidus (affichage uniquement des points) – Top gauche
    ax_time = fig.add_subplot(gs[0, 0])
    ax_time.scatter(np.arange(len(V_est)), V_est, color='green')
    ax_time.set_title("Série temporelle des résidus")
    ax_time.set_xlabel("Temps / Index")
    ax_time.set_ylabel("Résidus")
    
    # 2. Histogramme des résidus (nombre brut d'observations) – Top droite
    ax_hist = fig.add_subplot(gs[0, 1])
    sns.histplot(V_est, bins=30, stat="count", color='skyblue', edgecolor='black', ax=ax_hist)
    ax_hist.set_title("Histogramme des résidus")
    ax_hist.set_xlabel("Résidus")
    ax_hist.set_ylabel("Nombre d'observations")
    
    # 3. Q-Q Plot des résidus – Milieu gauche
    ax_qq = fig.add_subplot(gs[1, 0])
    sm.qqplot(V_est, line='s', ax=ax_qq)
    ax_qq.set_title("Q-Q Plot des résidus")
    
    # 4. Graphique des résidus vs. valeurs prédites – Milieu droite
    ax_scatter = fig.add_subplot(gs[1, 1])
    ax_scatter.scatter(B_est, V_est, alpha=0.7, color='darkorange')
    ax_scatter.axhline(0, color='red', linestyle='--')
    ax_scatter.set_xlabel("Valeurs prédites")
    ax_scatter.set_ylabel("Résidus")
    ax_scatter.set_title("Résidus vs. Valeurs prédites")
    
    # 5. Encadré avec les statistiques – Bas, sur 2 colonnes
    ax_stats = fig.add_subplot(gs[2, :])
    ax_stats.axis('off')  # Masquer les axes pour cet encart
    texte_stats = (
        f"Moyenne    : {moyenne:.4f}\n"
        f"Variance   : {variance:.4f}\n"
        f"Écart-type : {ecart_type:.4f}\n"
        f"Skewness   : {skewness:.4f}\n"
        f"Kurtosis   : {kurtosis:.4f}"
    )
    ax_stats.text(0.5, 0.5, texte_stats, transform=ax_stats.transAxes,
                  fontsize=14, verticalalignment='center', horizontalalignment='center',
                  bbox=dict(facecolor='wheat', edgecolor='black', boxstyle='round,pad=1'))
    ax_stats.set_title("Statistiques des résidus", fontsize=16)
    
    # Si un titre global est fourni, l'ajouter en haut de la figure
    if figure_title is not None:
        fig.suptitle(figure_title, fontsize=20)
        # Ajuster l'espacement pour ne pas superposer le titre aux sous-graphiques
        plt.subplots_adjust(top=0.92)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Sauvegarder la figure si un chemin de sauvegarde est fourni
    if save_path is not None:
        fig.savefig(save_path)
        print(f"Figure sauvegardée dans : {save_path}")
    
    plt.show()
    return fig
