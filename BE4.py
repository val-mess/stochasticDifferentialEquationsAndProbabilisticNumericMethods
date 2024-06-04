import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import random 
from scipy.stats import uniform
from scipy.spatial.distance import pdist


# Paramètres
niter = 10000
startval = 3
proposalsd = 0.5

def log_exp_target(x):
    if x>0:
        y=np.log(stats.expon.pdf(x))
    else:
      y=-1000000
    return y

# Initialisation de la liste x
x = np.zeros(niter)
x[0] = startval

# Boucle Metropolis-Hastings
for i in range(1, niter):
    currentx = x[i-1]
    proposedx = np.random.normal(loc=currentx, scale=proposalsd)
    if proposedx < 0:
        x[i] = currentx
    else:
        A =  np.exp(log_exp_target(proposedx) - log_exp_target(currentx))
        
        if np.random.rand() < A:
            x[i] = proposedx
        else:
            x[i] = currentx


# plt.hist(x, bins=50, density=True, alpha=0.75)
# plt.title('Histogramme des valeurs générées')
# plt.show()


def generer_d_uplet_points(d):
    points = [(random.random(), random.random()) for _ in range(d)]
    return points

def appliquer_loi_normale(point, variance):
    nouvelle_valeur = np.random.normal(loc=0, scale=variance)
    return point + nouvelle_valeur

def modifier_point_d_uplet_au_hasard(d_uplet, variance):
    indice_point_a_modifier = random.randint(0, len(d_uplet) - 1)
    d_uplet_modifie = list(d_uplet)
    d_uplet_modifie[indice_point_a_modifier] = tuple(appliquer_loi_normale(coord, variance) for coord in d_uplet[indice_point_a_modifier])
    return tuple(d_uplet_modifie), indice_point_a_modifier

def plot_d_uplet_points(points, indice_point_modifie=None):
    x, y = zip(*points)
    colors = ['blue' if i != indice_point_modifie else 'red' for i in range(len(points))]
    plt.scatter(x, y, color=colors)
    plt.xlabel('Axe X')
    plt.ylabel('Axe Y')
    plt.title('Points aléatoires dans [0, 1]^2')
    plt.show()

# Exemple pour d = 3
d=100
d_uplet = generer_d_uplet_points(d)
#print("D-uplet de points avant la modification:", d_uplet)

# Modifier un point au hasard en lui appliquant une loi normale
variance = 0.1
d_uplet_modifie, indice_point_modifie = modifier_point_d_uplet_au_hasard(d_uplet, variance)
#print("D-uplet de points après la modification:", d_uplet_modifie)

# Afficher les points avec la couleur différente pour le point modifié
# plt.figure()
# plot_d_uplet_points(d_uplet, indice_point_modifie)
# plt.figure()
# plot_d_uplet_points(d_uplet_modifie, indice_point_modifie)


# Fonction pour calculer le nombre de paires de points distants de moins de epsilon
def compter_paires_proches(points, epsilon):
    distances = pdist(points)
    return np.sum(distances < epsilon)

# Boucle Metropolis-Hastings
niter = 1000
x = [generer_d_uplet_points(d)]  # Initialisation de x comme une liste de points
variance = 0.1
epsilon = 0.4  # À ajuster selon vos besoins
gamma_constant = 0.9

for i in range(1, niter):
    currentx = x[-1]
    proposedx, _ = modifier_point_d_uplet_au_hasard(currentx, variance)

    # Compter le nombre de paires de points distants de moins de epsilon
    nb_paires_proches_current = compter_paires_proches(currentx, epsilon)
    nb_paires_proches_proposed = compter_paires_proches(proposedx, epsilon)

    # Calculer gamma comme une constante élevée à la puissance du nombre de paires
    gamma_current = np.power(gamma_constant, nb_paires_proches_current)
    gamma_proposed = np.power(gamma_constant, nb_paires_proches_proposed)

    # Générer des échantillons à partir de la distribution uniforme
    uniform_samples_current = [uniform.rvs() for _ in range(d)]
    uniform_samples_proposed = [uniform.rvs() for _ in range(d)]

    # Calculer la densité de la distribution uniforme sur [0, 1]^2^d
    uniform_density_current = np.prod(uniform.pdf(uniform_samples_current))
    uniform_density_proposed = np.prod(uniform.pdf(uniform_samples_proposed))

    # Calculer le rapport entre les densités
    A = min(1, (uniform_density_proposed * gamma_proposed) / (uniform_density_current * gamma_current))

    if np.random.rand() < A:
        x.append(proposedx)
    else:
        x.append(currentx)


# Tracer l'histogramme
plot_d_uplet_points(x[-1])