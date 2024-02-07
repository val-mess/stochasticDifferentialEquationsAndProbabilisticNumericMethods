#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 13:50:30 2023

@author: valentinmessina
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# =============================================================================
# Exercice 1 (Equation de Fisher-Wright-Diffusion Approximation) 
# =============================================================================

# Question 1

n=100
x=0.4
T=10

def pi_n(z):
    return stats.binom.rvs(n,z,size=1)/n

def Y(n):
    Y_n=np.zeros(n*T)
    Y_n[0]=x
    for i in range(n*T-1):
        Y_n[i+1]=pi_n(Y_n[i])
    return Y_n

def MC(M):
    esp=0
    var=0
    for i in range(M):
        esp+=np.mean(Y(n))
        var+=stats.tvar(Y(n))
    esp=esp/M
    var=var/M
    return [esp,var]

#print("E[Xt]=",MC(100)[0])
#print("Var[XT]=",MC(100)[1])

def sigma(x):
    return x

def sigmaPrime(x):
    return 1


U=np.array([[0,-1],[1,0]])
c=-0.5

def b(x):
    return c*x

def sigma2(x):
    return U*x

N=1000
T=1


def SchemaEuler(sigma,T,N,X0):
    X0=1
    X_N=np.zeros(N)
    X_N[0]=X0
    h=T/N
    for n in range(1,N):
        X_N[n]=X_N[n-1]+sigma(X_N[n-1])*np.sqrt(h)*stats.norm.rvs(0,1)
    return X_N
    
def SchemaMilstein(sigma,sigmaPrime,T,N,X0):
    X0=1
    X_N=np.zeros(N)
    X_N[0]=X0
    h=T/N
    for n in range(1,N):
        X_N[n]=X_N[n-1] \
        -0.5*sigmaPrime(X_N[n-1])*sigma(X_N[n-1])*h \
        +sigma(X_N[n-1])*np.sqrt(h)*stats.norm.rvs(0,1) \
        +0.5*sigmaPrime(X_N[n-1])*sigma(X_N[n-1])*h*(stats.norm.rvs(0,1)**2)
    return X_N

def SolExa(sigma,T,N,X0):
    X0=1
    X_N=np.zeros(N)
    X_N=X0
    h=T/N
    t=np.linspace(0,T,N)
    X_N=X0*np.exp(np.sqrt(h)*stats.norm.rvs(0,t,size=N)-t/2)
    return X_N
    

plt.grid()
plt.plot(np.linspace(0,T,N),SchemaEuler(sigma,T,N,1),label="Euler")
plt.plot(np.linspace(0,T,N),SchemaMilstein(sigma,sigmaPrime,T,N,1),label="Milstein")
plt.plot(np.linspace(0,T,N),SolExa(sigma,T,N,1),label="Exacte")
plt.xlabel("$t$")
plt.ylabel("$(X_t)_{t\geqslant 0}$")
plt.title("Simulation de $(X_t)_{t\geqslant 0}$")
plt.legend()

def Schemas(sigma,sigmaPrime,T,N,X0):
    X0=1
    X_N_Milstein=np.zeros(N)
    X_N_Euler=np.zeros(N)
    X_N_Exacte=np.zeros(N)
    X_N_Milstein[0]=X0
    X_N_Euler[0]=X0
    X_N_Exacte[0]=X0
    h=T/N
    DeltaBm = stats.norm.rvs(0,1,size=N)
    for n in range(1,N):
        X_N_Euler[n]=X_N_Euler[n-1]+sigma(X_N_Euler[n-1])*np.sqrt(h)*DeltaBm[n]
        
        X_N_Milstein[n]=X_N_Milstein[n-1] \
        -0.5*sigma(X_N_Milstein[n-1])*h \
        +sigma(X_N_Milstein[n-1])*np.sqrt(h)*DeltaBm[n] \
        +0.5*sigma(X_N_Milstein[n-1])*h*(DeltaBm[n]**2)
        
        
        X_N_Exacte[n]=X_N_Exacte[n-1]*np.exp(np.sqrt(h)*DeltaBm[n]-h/2)
        
        
    return X_N_Exacte,X_N_Euler,X_N_Milstein

plt.grid()
plt.plot(np.linspace(0,T,N),Schemas(sigma,sigmaPrime,T,N,1)[1],label="Euler")
plt.plot(np.linspace(0,T,N),Schemas(sigma,sigmaPrime,T,N,1)[2],label="Milstein")
plt.plot(np.linspace(0,T,N),Schemas(sigma,sigmaPrime,T,N,1)[0],label="Exacte")
plt.xlabel("$t$")
plt.ylabel("$(X_t)_{t\geqslant 0}$")
plt.title("Simulation de $(X_t)_{t\geqslant 0}$")
plt.legend()

# N=10000
# x=0.4
# X_N=np.zeros((N,2),dtype=float)
# X_N[0]=np.array([1,0])
# h=0.01
# for k in range(1):
#     for n in range(1,N):
#         X_N[n] = X_N[n-1] + b(X_N[n-1])*h + np.sqrt(h) * np.random.normal(0, 1, size=(2,))

#     plt.grid()
#     plt.plot(np.arange(N),X_N)



# Définir les fonctions sigma, sigmaPrime, U, c, b, sigma2, SchemaEuler, SchemaMilstein, SolExa


# Définir les fonctions sigma, sigmaPrime, U, c, b, sigma2, SchemaEuler, SchemaMilstein, SolExa

def erreur_mc_euler(M, sigma, T, N, X0):
    total_erreur = 0
    
    for _ in range(M):
        # Simulation du schéma d'Euler
        euler_simulation = SchemaEuler(sigma, T, N, X0)
        
        # Simulation de la solution exacte
        exact_solution = SolExa(sigma, T, N, X0)
        
        # Calcul de l'erreur quadratique pour cette simulation
        erreur = np.mean((euler_simulation - exact_solution)**2)
        
        # Ajouter l'erreur à la somme totale
        total_erreur += erreur
    
    # Calculer l'erreur moyenne sur toutes les simulations
    erreur_moyenne = total_erreur / M
    
    return erreur_moyenne

# Paramètres
M = 100  # Nombre d'itérations de la méthode de Monte Carlo
T = 1
X0 = 1

# Valeurs de N à tester (pas de 2 à 2^10)
valeurs_N = 2**np.arange(1, 11)

# Initialiser un tableau pour stocker les erreurs
erreurs = []

# Calculer l'erreur moyenne pour chaque valeur de N
for N in valeurs_N:
    erreur_moyenne = erreur_mc_euler(M, sigma, T, N, X0)
    erreurs.append(erreur_moyenne)

# # Tracer l'erreur en fonction de N sur une échelle logarithmique
# plt.loglog(1 / valeurs_N, erreurs, marker='o', linestyle='-', label='Erreur moyenne')

# # Tracer une ligne de référence O(h)
# plt.loglog(1 / valeurs_N, 1 / valeurs_N, linestyle='--', label='O(h)')

# # Étiqueter l'axe x et y
# plt.xlabel('Pas h (log scale)')
# plt.ylabel('Erreur moyenne (log scale)')

# # Ajouter une légende
# plt.legend()

# # Afficher le graphique
# plt.show()
