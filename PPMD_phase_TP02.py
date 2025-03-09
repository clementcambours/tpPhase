#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 16:44:03 2024

Filière ING3 - PPMD - Traitement de la mesure de phase

@author: Samuel Nahmani (1,2)
https://www.ipgp.fr/annuaire/nahmani/)
contact : nahmani@ipgp.fr ou samuel.nahmani@ign.fr
(1) Université Paris Cité, Institut de physique du globe de Paris, CNRS, IGN, F-75005 Paris, France.
(2) Univ Gustave Eiffel, ENSG, IGN, F-77455 Marne-la-Vallée, France. 

Version: 1.0
Dépendances: pandas, numpy, geodezyx, datetime, gpsdatetime, gnsstoolbox

"""

#%%
# GeodeZYX Toolbox’s
# [Sakic et al., 2019]
# Sakic, Pierre; Mansur, Gustavo; Chaiyaporn, Kitpracha; Ballu, Valérie (2019): 
# The geodeZYX toolbox: a versatile Python 3 toolbox for geodetic-oriented purposes. V. 4.0. 
# GFZ Data Services. http://doi.org/10.5880/GFZ.1.1.2019.002
#
# Documentation
# https://ipgp.github.io/geodezyx/getting_started.html
# 
# Installation / Réinstallation
# pip uninstall geodezyx
# pip install "geodezyx[full] @ git+https://github.com/IPGP/geodezyx"

#%%
# gpsdatetime
# pip install gpsdatetime
#
# Python GPS date/time management package
# Copyright (C) 2014-2023, Jacques Beilin / ENSG-Geomatique
# Distributed under terms of the CECILL-C licence.
#%%
# GnssToolbox - Python package for GNSS learning
# Copyright (C) 2014-2023, Jacques Beilin / ENSG-Geomatique
# Distributed under terms of the CECILL-C licence.
#
# pip install gnsstoolbox
#
#%%
# GeodeZYX Toolbox’s - [Sakic et al., 2019]
import geodezyx
import geodezyx.conv as conv                  # Import the conversion module
import datetime as dt

#
import gpsdatetime as gpst

import gnsstoolbox.orbits as orb
import gnsstoolbox.gnss_const as gnss_const
import gnsstoolbox.gnsstools as tools
import gnsstoolbox.gnss_process as proc
import gnsstoolbox.gnss_corr as gnss_corr

import pandas as pd
import numpy as np
from scipy.stats import norm

# pour visualiser les données
import matplotlib.pyplot as plt


# pour chercher des chaînes de caractère dans des fichiers
import re


def grep_file(pattern, filename):
    """ Recherche un motif dans un fichier et retourne les lignes correspondantes. """
    results = []
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            if re.search(pattern, line):
                results.append(line.strip())  # Supprime les espaces et les sauts de ligne

    return results  # Retourne les lignes trouvées

# Exemple d'utilisation
#grep_file(r"test", "mon_fichier.txt")

#%%
# Chargement des fichiers RINEX d'observation
#fichier_rnx='./data/data-2019/mlvl176z.18o'
fichier_rnx='/Users/clementcambours/Documents/ENSG-geomatique/ING3/phase/[GEOLOC] data GNSS/PPMD_2025/mlvl176z.18o'

# Chargement des données RINEX d'observation dans un pandas dataframe via  GeodeZYX
df_rnx = geodezyx.files_rw.read_rinex_obs(fichier_rnx, set_index=['epoch', 'prn'])

# nettoyage cf fin du TP01
df_rnx = df_rnx.dropna(axis=1, how='all')
rows_with_nan = df_rnx[['C1','L1', 'L2']].isna().any(axis=1)
df_removed = df_rnx[rows_with_nan]
df_rnx = df_rnx.dropna(subset=['C1','L1', 'L2'])
del rows_with_nan

# on ne garde que les mesures des satellites GPS
# Filtrer pour garder uniquement les lignes où la colonne 'sys' contient 'G'
df_filtre = df_rnx[df_rnx['sys'].str.contains('G')]

df_rnx = df_filtre
del df_filtre

df_rnx = df_rnx.dropna(axis=1, how='all')

# ajout de l'indice de ligne
df_rnx['ind_ligne'] = range(len(df_rnx)) 


# # Chargement des données RINEX d'observation via  GnssToolbox
# # Trop de lourdeur pour pas grand chose ... autant passer par un équivalent de grep
# import gnsstoolbox.rinex_o as rx
# my_rnx =  rx.rinex_o()
# my_rnx.loadRinexO(fichier_rnx) 

# # Position initiale du récepteur (à partir du header RINEX)
# P_rnx_header = np.array([my_rnx.headers[0].X, my_rnx.headers[0].Y, my_rnx.headers[0].Z])
# del my_rnx

columns =  grep_file(r"APPROX POSITION XYZ", fichier_rnx)
if columns:
    # Séparer la ligne en supprimant le texte "APPROX POSITION XYZ"
    valeurs = columns[0].split()[:3]  # Récupérer uniquement les 3 premières valeurs
    P_rnx_header = np.array(valeurs, dtype=float)  # Convertir en numpy array
    print("Coordonnées XYZ :", P_rnx_header)
else:
    print("Aucune correspondance trouvée.")


del columns, valeurs

#%%
# Chargement des fichiers d'orbites
fichier_sp3  = ['/Users/clementcambours/Documents/ENSG-geomatique/ING3/phase/[GEOLOC] data GNSS/PPMD_2025/igs20071.sp3', '/Users/clementcambours/Documents/ENSG-geomatique/ING3/phase/[GEOLOC] data GNSS/PPMD_2025/igs20072.sp3']
fichier_brdc = '/Users/clementcambours/Documents/ENSG-geomatique/ING3/phase/[GEOLOC] data GNSS/PPMD_2025/mlvl176z.18n'

mysp3 = orb.orbit()
mysp3.loadSp3(fichier_sp3)

mynav = orb.orbit()
mynav.loadRinexN('/Users/clementcambours/Documents/ENSG-geomatique/ING3/phase/[GEOLOC] data GNSS/PPMD_2025/mlvl176z.18n')

# Il faut calculer la position de chaque satellite GNSS à chaque temps d'émission
t = gpst.gpsdatetime()

X_sat = []
Y_sat = []
Z_sat = []
dte_sat = []
dRelat = []

for (time_i,prn_i) in df_rnx.index:

    t.rinex_t(time_i.to_pydatetime().strftime('%y %m %d %H %M %S.%f'))
    
    # temps d'émission - cas idéal mathématique
    t_emission_mjd  = t.mjd - df_rnx.loc[(time_i,prn_i), 'C1'] / gnss_const.c / 86400.0
    
    (X_sat_v,Y_sat_v,Z_sat_v,dte_sat_v)	 = mysp3.calcSatCoord(prn_i[0], int(prn_i[1:]),t_emission_mjd)
    
    # calcul de l'effet relativiste
    delta_t = 1e-3 # écart de temps en +/- pour calculer la dérivée 
    (Xs1,Ys1,Zs1,clocks1) = mysp3.calcSatCoord(prn_i[0], int(prn_i[1:]),t_emission_mjd - delta_t / 86400.0)    
    (Xs2,Ys2,Zs2,clocks2) = mysp3.calcSatCoord(prn_i[0], int(prn_i[1:]),t_emission_mjd + delta_t / 86400.0)  
    
    VX      = (np.array([Xs2-Xs1, Ys2-Ys1, Zs2-Zs1]))/2.0/delta_t
    VX0     = np.array([X_sat_v,Y_sat_v,Z_sat_v])

    dRelat_v  = -2.0 * VX0.T@ VX /(gnss_const.c **2)
    
    # temps d'emission du signal GNSS en temps GNSS (mjd)
    # le temps d'émission idéal doit être corrigé de l'erreur d'horloge satellite 
    # et de l'effet relativiste
    t_emission_mjd = t_emission_mjd - dte_sat_v / 86400.0 - dRelat_v / 86400.0
    
    # Recalcul de la position du satellite au temps d'emission (temps GNSS en mjd)   
    (X_sat_v,Y_sat_v,Z_sat_v,dte_sat_v)	 = mysp3.calcSatCoord(prn_i[0], int(prn_i[1:]),t_emission_mjd)
    
    
    X_sat.append(X_sat_v)
    Y_sat.append(Y_sat_v)
    Z_sat.append(Z_sat_v)
    dte_sat.append(dte_sat_v)
    dRelat.append(dRelat_v)
    
    
df_rnx['X_sat']   = X_sat 
df_rnx['Y_sat']   = Y_sat 
df_rnx['Z_sat']   = Z_sat
df_rnx['dte_sat'] = dte_sat
df_rnx['dRelat']  = dRelat


del X_sat, Y_sat, Z_sat, dte_sat, time_i, prn_i, t_emission_mjd, t, X_sat_v, Y_sat_v, Z_sat_v, dte_sat_v, dRelat_v, dRelat



df_rnx_new = pd.read_pickle('/Users/clementcambours/Documents/ENSG-geomatique/ING3/phase/[GEOLOC] data GNSS/df_rnx_new.pkl')
df_Sagnac_new = pd.read_pickle('/Users/clementcambours/Documents/ENSG-geomatique/ING3/phase/[GEOLOC] data GNSS/df_Sagnac_new.pkl')

C1 = df_rnx['C1'].values
C2 = (gnss_const.f1**2*df_rnx['C1'].values - gnss_const.f2**2*df_rnx['P2'].values)/(gnss_const.f1**2 - gnss_const.f2**2)
C3 = df_rnx_new['P3'].values
L3 = df_rnx_new['L3'].values * (gnss_const.f1**2 - gnss_const.f2**2) # on multiplie par la différence de fréquence pour avoir la longueur d'onde
Dobs2 = C1 + (df_rnx['dte_sat'].values + df_rnx['dRelat'].values) * gnss_const.c
Dobs3 = C2 + (df_rnx['dte_sat'].values + df_rnx['dRelat'].values) * gnss_const.c
Dobs4 = C3 + (df_rnx_new['dte_sat'].values + df_rnx_new['dRelat'].values) * gnss_const.c
Dobs5 = L3 + (df_rnx_new['dte_sat'].values + df_rnx_new['dRelat'].values) * gnss_const.c
mfh_ZHD = df_rnx_new['mfh']*df_rnx_new['ZHD']
Dobs6 = Dobs4 - mfh_ZHD

df_rnx_new["X_sat_rot"] = df_rnx_new["X_sat"]
df_rnx_new["Y_sat_rot"] = df_rnx_new["Y_sat"]
df_rnx_new["Z_sat_rot"] = df_rnx_new["Z_sat"]
for index, row in df_rnx_new.iterrows():
    T_vol = row['P3'] / gnss_const.c
    angle_rot = T_vol * np.pi * 2 / (86164.096)
    X, Y, Z = tools.toolRotZ(row['X_sat'], row['Y_sat'], row['Z_sat'], -angle_rot)
    df_rnx_new.at[index, 'X_sat_rot'] = X
    df_rnx_new.at[index, 'Y_sat_rot'] = Y
    df_rnx_new.at[index, 'Z_sat_rot'] = Z

Xsat = df_rnx_new['X_sat_rot'].values
Ysat = df_rnx_new['Y_sat_rot'].values
Zsat = df_rnx_new['Z_sat_rot'].values
PosSatSagn = np.column_stack((Xsat, Ysat, Zsat))


Xsat = df_rnx['X_sat'].values
Ysat = df_rnx['Y_sat'].values
Zsat = df_rnx['Z_sat'].values


PosSat = np.column_stack((Xsat, Ysat, Zsat))

# Initial guess for the parameters X, Y, Z
X0 = np.array([[0], [0], [0]])
X1 = np.array([[0], [0], [0], [0]])

def create_matB(Dobs, PosSat, X0):
    """Create the B matrix for the Gauss-Newton method."""
    Pcorr = Dobs.reshape(-1, 1) 
    Xsat = PosSat[:, 0].reshape(-1, 1)
    Ysat = PosSat[:, 1].reshape(-1, 1)
    Zsat = PosSat[:, 2].reshape(-1, 1)

    if X0.shape[0] == 3:
        Pcorr0 = np.sqrt((X0[0][0] - Xsat)**2 + (X0[1][0] - Ysat)**2 + (X0[2][0] - Zsat)**2)
    elif X0.shape[0] == 4:
        Pcorr0 = np.sqrt((X0[0][0] - Xsat)**2 + (X0[1][0] - Ysat)**2 + (X0[2][0] - Zsat)**2) + X0[3][0]
    B = Pcorr - Pcorr0
    return B

def create_matA(Dobs, PosSat, X0, nbre_param):
    """Create the A matrix for the Gauss-Newton method."""
    Xsat = PosSat[:, 0]
    Ysat = PosSat[:, 1]
    Zsat = PosSat[:, 2]
    
    A = np.ones((Dobs.shape[0], nbre_param))
    A[:, 0] = (X0[0][0] - Xsat) / np.sqrt((X0[0][0] - Xsat)**2 + (X0[1][0] - Ysat)**2 + (X0[2][0] - Zsat)**2)
    A[:, 1] = (X0[1][0] - Ysat) / np.sqrt((X0[0][0] - Xsat)**2 + (X0[1][0] - Ysat)**2 + (X0[2][0] - Zsat)**2)
    A[:, 2] = (X0[2][0] - Zsat) / np.sqrt((X0[0][0] - Xsat)**2 + (X0[1][0] - Ysat)**2 + (X0[2][0] - Zsat)**2)
    return A

def create_matP(n, err):
    """Create the weight matrix P."""
    P = np.eye(n) * err
    return P

def MoindreCarre(Dobs, PosSat, X0, nbre_param):
    """Perform the least squares estimation using the Gauss-Newton method."""
    matA = create_matA(Dobs, PosSat, X0, nbre_param)
    matB = create_matB(Dobs, PosSat, X0)
    P = create_matP(Dobs.shape[0], err=1)
    
    dX = np.linalg.inv(matA.T @ P @ matA) @ matA.T @ P @ matB
    X_chap = X0 + dX
    Vchap = matB - matA @ dX
    
    n = matA.shape[0]
    p = matA.shape[1]
    sigma02_old = 0
    sigma02 = (Vchap.T @ P @ Vchap)[0, 0] / float(n - p)
    
    varVchap = sigma02 * (np.linalg.inv(P) - matA @ np.linalg.inv(matA.T @ P @ matA) @ matA.T)
    Vnor = np.linalg.inv(np.sqrt(np.diag(np.diag(varVchap)))) @ Vchap
    
    cpt = 0
    print('Iteration : ', cpt)
    
    while np.abs(sigma02 - sigma02_old) > 1e-6:
        X0 = X_chap
        sigma02_old = sigma02
        matA = create_matA(Dobs, PosSat, X0, nbre_param)
        matB = create_matB(Dobs, PosSat, X0)
        
        n = matA.shape[0]
        p = matA.shape[1]
        dX = np.linalg.inv(matA.T @ P @ matA) @ matA.T @ P @ matB
        X_chap = X0 + dX
        Vchap = matB - matA @ dX
        sigma02 = (Vchap.T @ P @ Vchap)[0, 0] / float(n - p)
        
        varVchap = sigma02 * (np.linalg.inv(P) - matA @ np.linalg.inv(matA.T @ P @ matA) @ matA.T)
        Vnor = np.linalg.inv(np.sqrt(np.diag(np.diag(varVchap)))) @ Vchap
        
        cpt += 1
        print('Iteration : ', cpt)
    
    SigmaX = sigma02 * np.linalg.inv(matA.T @ P @ matA)
    return X_chap, Vchap, sigma02, Vnor, SigmaX

# Calcul trivial de la distance entre la position estimée et la position initiale sans correction
X_chap, Vchap, sigma02, Vnor, SigmaX = MoindreCarre(Dobs=C1, PosSat=PosSat, X0=X0, nbre_param=3)

# Plot the results
plt.figure(figsize=(12, 6))

# Histogram of Vnor with Gaussian fit
plt.subplot(1, 2, 1)
plt.hist(Vnor, bins=30, density=True, alpha=0.6, color='b', label='Vnor Histogram')
mu, std = norm.fit(Vnor)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2, label='Gaussian Fit')
plt.title('Histogram of Vnor with Gaussian Fit')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()

# Scatter plot of Vchap
plt.subplot(1, 2, 2)
plt.scatter(range(len(Vchap)), Vchap, color='r', s=5, label='Vchap Points')
plt.title('Vchap Values')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()

plt.tight_layout()
plt.show()

# Display the estimated parameters
print(f"Estimated parameters (X, Y, Z): {X_chap.flatten()}")
# %%
# Calcul avec estimation erreur horloge recepteur sur 1 heure
X_chap, Vchap, sigma02, Vnor, SigmaX = MoindreCarre(Dobs=Dobs2, PosSat=PosSat, X0=X1, nbre_param=4)

# Plot the results
plt.figure(figsize=(12, 6))

# Histogram of Vnor with Gaussian fit
plt.subplot(1, 2, 1)
plt.hist(Vnor, bins=30, density=True, alpha=0.6, color='b', label='Vnor Histogram with dte_sat')
mu, std = norm.fit(Vnor)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2, label='Gaussian Fit')
plt.title('Histogram of Vnor with Gaussian Fit with dte_sat')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()

# Scatter plot of Vchap
plt.subplot(1, 2, 2)
plt.scatter(range(len(Vchap)), Vchap, color='r', s=5, label='Vchap Points')
plt.title('Vchap Values')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()

plt.tight_layout()
plt.show()

# Display the estimated parameters
print(f"Estimated parameters (X, Y, Z, dte_sat): {X_chap.flatten()}")
# %%


dist_P_est_rnx_header = np.sqrt((np.sum((X_chap - P_rnx_header)**2)))
E, N, U = tools.toolCartLocGRS80(P_rnx_header[0], P_rnx_header[1], P_rnx_header[2], X_chap[0][0], X_chap[1][0], X_chap[2][0])
print(f"Easting : {E}, Northing : {N}, Up : {U}")
# %%

########### Calcul avec estimation erreur horloge recepteur par epoque

def CmatA(Dobs, PosSat, X0, epochs):
    """Create the A matrix for the Gauss-Newton method."""
    Xsat = PosSat[:, 0]
    Ysat = PosSat[:, 1]
    Zsat = PosSat[:, 2]
    
    A = np.zeros((Dobs.shape[0], 3))
    denom = np.sqrt((X0[0] - Xsat)**2 + (X0[1] - Ysat)**2 + (X0[2] - Zsat)**2)
    A[:, 0] = (X0[0] - Xsat) / denom
    A[:, 1] = (X0[1] - Ysat) / denom
    A[:, 2] = (X0[2] - Zsat) / denom
    
    nA = np.zeros((len(df_rnx), 3 + len(epochs)))
    nA[:, :3] = A
    A = nA
    for e in range(len(epochs)):
            A[:, 3 + e][df_rnx.index.get_level_values('epoch') == epochs[e]] = 1  
    return A

def CmatB(Dobs, PosSat, X0):
    """Create the B matrix for the Gauss-Newton method."""
    Pcorr = Dobs.reshape(-1, 1) 

    Xsat = PosSat[:, 0].reshape(-1, 1)
    Ysat = PosSat[:, 1].reshape(-1, 1)
    Zsat = PosSat[:, 2].reshape(-1, 1)
    
    Pcorr0 = np.sqrt((X0[0][0] - Xsat)**2 + (X0[1][0] - Ysat)**2 + (X0[2][0] - Zsat)**2)
    
    B = Pcorr - Pcorr0
    return B

def create_matP(n, err):
    """Create the weight matrix P."""
    P = np.eye(n) * err
    return P

def MoindreCarre2(Dobs, PosSat, X0, epochs):
    """Perform the least squares estimation using the Gauss-Newton method."""



    matA = CmatA(Dobs, PosSat, X0, epochs)
    matB = CmatB(Dobs, PosSat, X0)
    P = create_matP(Dobs.shape[0], err=1)
    
    # Ajout d'une régularisation pour éviter une matrice singulière
    # regularization = 1e-6 * np.eye(matA.shape[1])
    # dX = np.linalg.inv(matA.T @ P @ matA + regularization) @ matA.T @ P @ matB
    dX = np.linalg.inv(matA.T @ P @ matA ) @ matA.T @ P @ matB
    X_chap = X0 + dX
    Vchap = matB - matA @ dX
    
    n = matA.shape[0]
    p = matA.shape[1]
    sigma02_old = 0
    sigma02 = (Vchap.T @ P @ Vchap)[0, 0] / float(n - p)
    
    # varVchap = sigma02 * (np.linalg.inv(P) - matA @ np.linalg.inv(matA.T @ P @ matA + regularization) @ matA.T)
    varVchap = sigma02 * (np.linalg.inv(P) - matA @ np.linalg.inv(matA.T @ P @ matA ) @ matA.T)
    Vnor = np.linalg.inv(np.sqrt(np.diag(np.diag(varVchap)))) @ Vchap
    
    cpt = 0
    print('Iteration : ', cpt)


    
    while np.abs(sigma02 - sigma02_old) > 1e-6:
        X0 = X_chap
        sigma02_old = sigma02
        matA = CmatA(Dobs, PosSat, X0, epochs)
        matB = CmatB(Dobs, PosSat, X0)
        
        n = matA.shape[0]
        p = matA.shape[1]
        # dX = np.linalg.inv(matA.T @ P @ matA + regularization) @ matA.T @ P @ matB
        dX = np.linalg.inv(matA.T @ P @ matA ) @ matA.T @ P @ matB
        X_chap = X0 + dX
        Vchap = matB - matA @ dX
        sigma02 = (Vchap.T @ P @ Vchap)[0, 0] / float(n - p)
        
        # varVchap = sigma02 * (np.linalg.inv(P) - matA @ np.linalg.inv(matA.T @ P @ matA + regularization) @ matA.T)
        varVchap = sigma02 * (np.linalg.inv(P) - matA @ np.linalg.inv(matA.T @ P @ matA ) @ matA.T)
        Vnor = np.linalg.inv(np.sqrt(np.diag(np.diag(varVchap)))) @ Vchap
        
        cpt += 1
        print('Iteration : ', cpt)
    
    # SigmaX = sigma02 * np.linalg.inv(matA.T @ P @ matA + regularization)
    SigmaX = sigma02 * np.linalg.inv(matA.T @ P @ matA )
    return X_chap, Vchap, sigma02, Vnor, SigmaX

# Exemple d'utilisation
Dobs2 = C1 + (df_rnx['dte_sat'].values + df_rnx['dRelat'].values) * gnss_const.c
epochs = np.unique(df_rnx.index.get_level_values('epoch'))
print(f"Nombre d'époques : {len(epochs)}")
Xsat = df_rnx['X_sat'].values
Ysat = df_rnx['Y_sat'].values
Zsat = df_rnx['Z_sat'].values
PosSat = np.column_stack((Xsat, Ysat, Zsat))

# Assurez-vous que X0 est correctement initialisé
X0 = np.zeros((len(epochs)+3, 1))  # Exemple d'initialisation

X_chap, Vchap, sigma02, Vnor, SigmaX = MoindreCarre2(Dobs=Dobs2, PosSat=PosSat, X0=X0, epochs=epochs)

# %%
# Plot the results
plt.figure(figsize=(12, 6))

# Histogram of Vnor with Gaussian fit
plt.subplot(1, 2, 1)
plt.hist(Vnor, bins=30, density=True, alpha=0.6, color='b', label='Vnor Histogram with dte_sat')
mu, std = norm.fit(Vnor)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2, label='Gaussian Fit')
plt.title('Histogram of Vnor with Gaussian Fit with dte_sat')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()

# Scatter plot of Vchap
plt.subplot(1, 2, 2)
plt.scatter(range(len(Vchap)), Vchap, color='r', s=5, label='Vchap Points')
plt.title('Vchap Values')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()

plt.tight_layout()
plt.show()

# Display the estimated parameters
print(f"Estimated parameters (X, Y, Z, dte_sat): {X_chap.flatten()}")
# %%
dist_P_est_rnx_header = np.sqrt((np.sum((X_chap - P_rnx_header)**2)))
E, N, U = tools.toolCartLocGRS80(P_rnx_header[0], P_rnx_header[1], P_rnx_header[2], X_chap[0][0], X_chap[1][0], X_chap[2][0])
print(f"Easting : {E}, Northing : {N}, Up : {U}")
# %%


############ Gestion effet sagnac 


X_chap, Vchap, sigma02, Vnor, SigmaX = MoindreCarre2(Dobs=Dobs2, PosSat=PosSatSagn, X0=X0, epochs=epochs)

plt.figure(figsize=(12, 6))

# Histogram of Vnor with Gaussian fit
plt.subplot(1, 2, 1)
plt.hist(Vnor, bins=30, density=True, alpha=0.6, color='b', label='Vnor Histogram with dte_sat')
mu, std = norm.fit(Vnor)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2, label='Gaussian Fit')
plt.title('Histogram of Vnor with Gaussian Fit with dte_sat')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()

# Scatter plot of Vchap
plt.subplot(1, 2, 2)
plt.scatter(range(len(Vchap)), Vchap, color='r', s=5, label='Vchap Points')
plt.title('Vchap Values')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()

plt.tight_layout()
plt.show()

# Display the estimated parameters
print(f"Estimated parameters (X, Y, Z, dte_sat): {X_chap.flatten()}")

dist_P_est_rnx_header = np.sqrt((np.sum((X_chap - P_rnx_header)**2)))
E, N, U = tools.toolCartLocGRS80(P_rnx_header[0], P_rnx_header[1], P_rnx_header[2], X_chap[0][0], X_chap[1][0], X_chap[2][0])
print(f"Easting : {E}, Northing : {N}, Up : {U}")
print(f"Distance entre la position estimée et la position initiale : {dist_P_est_rnx_header}")

# %%
################## Iono free

epochs = np.unique(df_rnx.index.get_level_values('epoch'))
print(f"Nombre d'époques : {len(epochs)}")
Xsat = df_rnx['X_sat_rot'].values
Ysat = df_rnx['Y_sat_rot'].values
Zsat = df_rnx['Z_sat_rot'].values
PosSat = np.column_stack((Xsat, Ysat, Zsat))

# Assurez-vous que X0 est correctement initialisé
X0 = np.zeros((len(epochs)+3, 1))  # Exemple d'initialisation

X_chap, Vchap, sigma02, Vnor, SigmaX = MoindreCarre2(Dobs=Dobs3, PosSat=PosSat, X0=X0, epochs=epochs)
print(f"Estimated parameters (X, Y, Z, dte_sat): {X_chap.flatten()}")
# %%

plt.figure(figsize=(12, 6))

# Histogram of Vnor with Gaussian fit
plt.subplot(1, 2, 1)
plt.hist(Vnor, bins=30, density=True, alpha=0.6, color='b', label='Vnor Histogram with dte_sat')
mu, std = norm.fit(Vnor)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2, label='Gaussian Fit')
plt.title('Histogram of Vnor with Gaussian Fit with dte_sat')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()

# Scatter plot of Vchap
plt.subplot(1, 2, 2)
plt.scatter(range(len(Vchap)), Vchap, color='r', s=5, label='Vchap Points')
plt.title('Vchap Values')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()

plt.tight_layout()
plt.show()

# Display the estimated parameters
print(f"Estimated parameters (X, Y, Z, dte_sat): {X_chap.flatten()}")

dist_P_est_rnx_header = np.sqrt((np.sum((X_chap - P_rnx_header)**2)))
E, N, U = tools.toolCartLocGRS80(P_rnx_header[0], P_rnx_header[1], P_rnx_header[2], X_chap[0][0], X_chap[1][0], X_chap[2][0])
print(f"Easting : {E}, Northing : {N}, Up : {U}")
print(f"Distance entre la position estimée et la position initiale : {dist_P_est_rnx_header}")

# %%


######### correction tropo


def CmatA(Dobs, PosSat, X0, epochs):
    """Create the A matrix for the Gauss-Newton method."""
    Xsat = PosSat[:, 0]
    Ysat = PosSat[:, 1]
    Zsat = PosSat[:, 2]
    
    A = np.zeros((Dobs.shape[0], 3))
    denom = np.sqrt((X0[0] - Xsat)**2 + (X0[1] - Ysat)**2 + (X0[2] - Zsat)**2)
    A[:, 0] = (X0[0] - Xsat) / denom
    A[:, 1] = (X0[1] - Ysat) / denom
    A[:, 2] = (X0[2] - Zsat) / denom
    
    nA = np.zeros((len(df_rnx), 3 + len(epochs)))
    nA[:, :3] = A
    A = nA
    for e in range(len(epochs)):
            A[:, 3 + e][df_rnx.index.get_level_values('epoch') == epochs[e]] = 1 
    _, elev = tools.toolAzEle(P_rnx_header[0], P_rnx_header[1], P_rnx_header[2], Xsat, Ysat, Zsat)
    A = np.concatenate((A, (1/np.sin(elev)).reshape(-1, 1)), axis=1)
    return A

def CmatB(Dobs, PosSat, X0):
    """Create the B matrix for the Gauss-Newton method."""
    Pcorr = Dobs.reshape(-1, 1) 

    Xsat = PosSat[:, 0].reshape(-1, 1)
    Ysat = PosSat[:, 1].reshape(-1, 1)
    Zsat = PosSat[:, 2].reshape(-1, 1)
    
    Pcorr0 = np.sqrt((X0[0][0] - Xsat)**2 + (X0[1][0] - Ysat)**2 + (X0[2][0] - Zsat)**2)
    
    B = Pcorr - Pcorr0
    return B

def create_matP(n, err):
    """Create the weight matrix P."""
    P = np.eye(n) * err
    return P

def MoindreCarre2(Dobs, PosSat, X0, epochs):
    """Perform the least squares estimation using the Gauss-Newton method."""



    matA = CmatA(Dobs, PosSat, X0, epochs)
    matB = CmatB(Dobs, PosSat, X0)
    P = create_matP(Dobs.shape[0], err=1)
    
    # Ajout d'une régularisation pour éviter une matrice singulière
    # regularization = 1e-6 * np.eye(matA.shape[1])
    # dX = np.linalg.inv(matA.T @ P @ matA + regularization) @ matA.T @ P @ matB
    dX = np.linalg.inv(matA.T @ P @ matA ) @ matA.T @ P @ matB
    X_chap = X0 + dX
    Vchap = matB - matA @ dX
    
    n = matA.shape[0]
    p = matA.shape[1]
    sigma02_old = 0
    sigma02 = (Vchap.T @ P @ Vchap)[0, 0] / float(n - p)
    
    # varVchap = sigma02 * (np.linalg.inv(P) - matA @ np.linalg.inv(matA.T @ P @ matA + regularization) @ matA.T)
    varVchap = sigma02 * (np.linalg.inv(P) - matA @ np.linalg.inv(matA.T @ P @ matA ) @ matA.T)
    Vnor = np.linalg.inv(np.sqrt(np.diag(np.diag(varVchap)))) @ Vchap
    
    cpt = 0
    print('Iteration : ', cpt)


    
    while np.abs(sigma02 - sigma02_old) > 1e-6:
        X0 = X_chap
        sigma02_old = sigma02
        matA = CmatA(Dobs, PosSat, X0, epochs)
        matB = CmatB(Dobs, PosSat, X0)
        
        n = matA.shape[0]
        p = matA.shape[1]
        # dX = np.linalg.inv(matA.T @ P @ matA + regularization) @ matA.T @ P @ matB
        dX = np.linalg.inv(matA.T @ P @ matA ) @ matA.T @ P @ matB
        X_chap = X0 + dX
        Vchap = matB - matA @ dX
        sigma02 = (Vchap.T @ P @ Vchap)[0, 0] / float(n - p)
        
        # varVchap = sigma02 * (np.linalg.inv(P) - matA @ np.linalg.inv(matA.T @ P @ matA + regularization) @ matA.T)
        varVchap = sigma02 * (np.linalg.inv(P) - matA @ np.linalg.inv(matA.T @ P @ matA ) @ matA.T)
        Vnor = np.linalg.inv(np.sqrt(np.diag(np.diag(varVchap)))) @ Vchap
        
        cpt += 1
        print('Iteration : ', cpt)
    
    # SigmaX = sigma02 * np.linalg.inv(matA.T @ P @ matA + regularization)
    SigmaX = sigma02 * np.linalg.inv(matA.T @ P @ matA )
    return X_chap, Vchap, sigma02, Vnor, SigmaX


X0 = np.zeros((len(epochs)+4, 1))  
X_chap, Vchap, sigma02, Vnor, SigmaX = MoindreCarre2(Dobs=Dobs3, PosSat=PosSatSagn, X0=X0, epochs=epochs)
print(f"Estimated parameters (X, Y, Z, dte_sat): {X_chap.flatten()}")

# %%
plt.figure(figsize=(12, 6))

# Histogram of Vnor with Gaussian fit
plt.subplot(1, 2, 1)
plt.hist(Vnor, bins=30, density=True, alpha=0.6, color='b', label='Vnor Histogram with dte_sat')
mu, std = norm.fit(Vnor)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2, label='Gaussian Fit')
plt.title('Histogram of Vnor with Gaussian Fit with dte_sat')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()

# Scatter plot of Vchap
plt.subplot(1, 2, 2)
plt.scatter(range(len(Vchap)), Vchap, color='r', s=5, label='Vchap Points')
plt.title('Vchap Values')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()

plt.tight_layout()
plt.show()

dist_P_est_rnx_header = np.sqrt((np.sum((X_chap - P_rnx_header)**2)))
E, N, U = tools.toolCartLocGRS80(P_rnx_header[0], P_rnx_header[1], P_rnx_header[2], X_chap[0][0], X_chap[1][0], X_chap[2][0])
print(f"Easting : {E}, Northing : {N}, Up : {U}")
print(f"Distance entre la position estimée et la position initiale : {dist_P_est_rnx_header}")

# %%

################## Calcul sur le code avec la iono-free avec le nouveau dataframe


# def CmatA(Dobs, PosSat, X0, epochs, prn):
#     """Create the A matrix for the Gauss-Newton method."""
#     Xsat = PosSat[:, 0]
#     Ysat = PosSat[:, 1]
#     Zsat = PosSat[:, 2]
    
#     A = np.zeros((Dobs.shape[0], 3))
#     denom = np.sqrt((X0[0] - Xsat)**2 + (X0[1] - Ysat)**2 + (X0[2] - Zsat)**2)
#     A[:, 0] = (X0[0] - Xsat) / denom
#     A[:, 1] = (X0[1] - Ysat) / denom
#     A[:, 2] = (X0[2] - Zsat) / denom
    
#     nA = np.zeros((len(df_rnx), 3 + len(epochs)))
#     nA[:, :3] = A
#     A = nA
#     for e in range(len(epochs)):
#         A[:, 3 + e][df_rnx.index.get_level_values('epoch') == epochs[e]] = 1 
#     _, elev = tools.toolAzEle(P_rnx_header[0], P_rnx_header[1], P_rnx_header[2], Xsat, Ysat, Zsat)
#     A = np.concatenate((A, (1/np.sin(elev)).reshape(-1, 1)), axis=1)
    
#     for p in range(len(prn)):
#         A[:, 3 + len(epochs) + p][df_rnx.index.get_level_values('prn') == prn[p]] = 1
#     return A


def CmatA(Dobs, PosSat, X0, epochs, tropo, MFW, prn=None):
    """Create the A matrix for the Gauss-Newton method."""
    Xsat = PosSat[:, 0]
    Ysat = PosSat[:, 1]
    Zsat = PosSat[:, 2]
    
    # Initialisation de A avec les 3 premières colonnes pour X, Y, Z
    A = np.zeros((Dobs.shape[0], 3))
    denom = np.sqrt((X0[0] - Xsat)**2 + (X0[1] - Ysat)**2 + (X0[2] - Zsat)**2)
    A[:, 0] = (X0[0] - Xsat) / denom
    A[:, 1] = (X0[1] - Ysat) / denom
    A[:, 2] = (X0[2] - Zsat) / denom
    
    # Ajout des colonnes pour les époques
    nA = np.zeros((Dobs.shape[0], 3 + len(epochs)))
    nA[:, :3] = A
    A = nA
    
    # Remplissage des colonnes pour les époques
    for e in range(len(epochs)):
        A[:, 3 + e][df_rnx_new.index.get_level_values('epoch') == epochs[e]] = 1 
    
    # Ajout de la colonne pour l'élévation
    if tropo :
        A = np.concatenate((A, MFW.reshape(-1, 1)), axis=1)
    
    # # Redimensionnement de A pour inclure les colonnes des PRN
    # A = np.concatenate((A, np.zeros((Dobs.shape[0], len(prn)))), axis=1)


    
    # # Remplissage des colonnes pour les PRN
    # for p in range(len(prn)):
    #     A[:, 3 + len(epochs) + 1 + p][df_rnx.index.get_level_values('prn') == prn[p]] = 1
    
    return A

def CmatB(Dobs, PosSat, X0):
    """Create the B matrix for the Gauss-Newton method."""
    Pcorr = Dobs.reshape(-1, 1) 

    Xsat = PosSat[:, 0].reshape(-1, 1)
    Ysat = PosSat[:, 1].reshape(-1, 1)
    Zsat = PosSat[:, 2].reshape(-1, 1)
    
    Pcorr0 = np.sqrt((X0[0][0] - Xsat)**2 + (X0[1][0] - Ysat)**2 + (X0[2][0] - Zsat)**2)
    
    
    B = Pcorr - Pcorr0 
    return B

def create_matP(n, err):
    """Create the weight matrix P."""
    P = np.eye(n) * err
    return P

def MoindreCarre2(Dobs, PosSat, X0, epochs, tropo, MFW, prn=None ):
    """Perform the least squares estimation using the Gauss-Newton method."""

    matA = CmatA(Dobs, PosSat, X0, epochs, tropo, MFW, prn=None)
    matB = CmatB(Dobs, PosSat, X0)
    P = create_matP(Dobs.shape[0], err=1)
    
    dX = np.linalg.inv(matA.T @ P @ matA ) @ matA.T @ P @ matB
    X_chap = X0 + dX
    Vchap = matB - matA @ dX
    
    n = matA.shape[0]
    p = matA.shape[1]
    sigma02_old = 0
    sigma02 = (Vchap.T @ P @ Vchap)[0, 0] / float(n - p)
    
    # varVchap = sigma02 * (np.linalg.inv(P) - matA @ np.linalg.inv(matA.T @ P @ matA + regularization) @ matA.T)
    varVchap = sigma02 * (np.linalg.inv(P) - matA @ np.linalg.inv(matA.T @ P @ matA ) @ matA.T)
    Vnor = np.linalg.inv(np.sqrt(np.diag(np.diag(varVchap)))) @ Vchap
    
    cpt = 0
    print('Iteration : ', cpt)


    
    while np.abs(sigma02 - sigma02_old) > 1e-6:
        X0 = X_chap
        sigma02_old = sigma02
        matA = CmatA(Dobs, PosSat, X0, epochs, tropo, MFW, prn=None)
        matB = CmatB(Dobs, PosSat, X0)
        
        n = matA.shape[0]
        p = matA.shape[1]
        # dX = np.linalg.inv(matA.T @ P @ matA + regularization) @ matA.T @ P @ matB
        dX = np.linalg.inv(matA.T @ P @ matA ) @ matA.T @ P @ matB
        X_chap = X0 + dX
        Vchap = matB - matA @ dX
        sigma02 = (Vchap.T @ P @ Vchap)[0, 0] / float(n - p)
        
        # varVchap = sigma02 * (np.linalg.inv(P) - matA @ np.linalg.inv(matA.T @ P @ matA + regularization) @ matA.T)
        varVchap = sigma02 * (np.linalg.inv(P) - matA @ np.linalg.inv(matA.T @ P @ matA ) @ matA.T)
        Vnor = np.linalg.inv(np.sqrt(np.diag(np.diag(varVchap)))) @ Vchap
        
        cpt += 1
        print('Iteration : ', cpt)
    
    # SigmaX = sigma02 * np.linalg.inv(matA.T @ P @ matA + regularization)
    SigmaX = sigma02 * np.linalg.inv(matA.T @ P @ matA )
    return X_chap, Vchap, sigma02, Vnor, SigmaX


# prn = np.unique(df_rnx_new.index.get_level_values('prn'))
MFW = df_rnx_new['mfw'].values
# X0 = np.zeros((len(epochs) + 4 + len(prn), 1))  
epochs = np.unique(df_rnx_new.index.get_level_values('epoch'))
print(f"Nombre d'époques : {len(epochs)}")
print(f"dobs4 taille : {Dobs4.shape}")
X0 = np.zeros((len(epochs) + 4, 1))
X_chap, Vchap, sigma02, Vnor, SigmaX = MoindreCarre2(Dobs=Dobs4, PosSat=PosSatSagn, X0=X0, epochs=epochs, tropo=True, MFW=MFW, prn=None)
print(f"Estimated parameters (X, Y, Z, dte_sat): {X_chap.flatten()}")
# %%

dist_P_est_rnx_header = np.sqrt((np.sum((X_chap - P_rnx_header)**2)))
E, N, U = tools.toolCartLocGRS80(P_rnx_header[0], P_rnx_header[1], P_rnx_header[2], X_chap[0][0], X_chap[1][0], X_chap[2][0])
print(f"Easting : {E}, Northing : {N}, Up : {U}")
print(f"Distance entre la position estimée et la position initiale : {dist_P_est_rnx_header}")

# %%
plt.figure(figsize=(12, 6))

# Histogram of Vnor with Gaussian fit
plt.subplot(1, 2, 1)
plt.hist(Vnor, bins=30, density=True, alpha=0.6, color='b', label='Vnor Histogram with dte_sat')
mu, std = norm.fit(Vnor)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2, label='Gaussian Fit')
plt.title('Histogram of Vnor with Gaussian Fit with dte_sat')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()

# Scatter plot of Vchap
plt.subplot(1, 2, 2)
plt.scatter(range(len(Vchap)), Vchap, color='r', s=5, label='Vchap Points')
plt.title('Vchap Values')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()

plt.tight_layout()
plt.show()

dist_P_est_rnx_header = np.sqrt((np.sum((X_chap - P_rnx_header)**2)))
E, N, U = tools.toolCartLocGRS80(P_rnx_header[0], P_rnx_header[1], P_rnx_header[2], X_chap[0][0], X_chap[1][0], X_chap[2][0])
print(f"Easting : {E}, Northing : {N}, Up : {U}")
print(f"Distance entre la position estimée et la position initiale : {dist_P_est_rnx_header}")


# %%
################### Double différence 

