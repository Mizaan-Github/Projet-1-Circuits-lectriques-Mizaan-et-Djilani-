import numpy as np

# Initialisation des données (CONSTANTES)

# Resistances (en Ohm)

R1 = 2200
R2 = 9700
R3 = 1700
R4 = 4300
R5 = 7700
R6 = 900
R7 = 2500
R8 = 9300
R9 = 9200
R10 = 3200

# Tension (Volt)

V = 100

# Matrice à gauche de l'egalite

R = np.array(
    [[1, 0, -1, -1, 0, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, -1, 0, -1, 0, 0, 0],
     [0, 0, 1, 0, 1, -1, 0, -1, 0, 0],
     [0, 0, 0, 0, 0, 0, -1, -1, 0, 1],
     [0, 0, 0, -1, 0, -1, 0, 0, 1, 0],
     [-R1, R2, -R3, 0, R5, 0, 0, 0, 0, 0],
     [0, 0, R3, -R4, 0, R6, 0, 0, 0, 0],
     [0, 0, 0, 0, -R5, 0, R7, -R8, 0, 0],
     [0, 0, 0, 0, 0, -R6, 0, R8, -R9, R10],
     [0, R2, 0, 0, 0, 0, R7, 0, 0, R10]],
    dtype=float)

# Vecteur b à droite de l'égalité.

b = np.zeros((R.shape[0], 1))
b[R.shape[0]-1, 0] = V

# Vecteur solution

x = np.zeros(R.shape[1])


def trouveMaxi(A: np.ndarray, k: int):  # Trouver le maximum
    """Trouve p entre k et m tel que |A[p][k]| soit max 

    Arguments:
        A (np.ndarray): La matrice initiale
        k (int): Le numero de colonne

    Returns:
        int: l'element de plus grande valeur absolue dans la colonne k sous la diagonale
    """
    maxi = abs(A[k, k])  # Haut gauche
    p = k
    m = A.shape[0]
    for i in range(k, m):
        absolute = abs(A[i, k])
        if maxi < absolute:  # Si nouveau max trouvé
            maxi = absolute
            p = i
    return p


# Algorithme de Gauss-Jordan optimisée partiellement
def gaussOptiPartielle(A: np.ndarray, precision=0.000001):
    """ Elimination de Gauss-Jordan avec optimisation partielle

    Arguments:
        A (np.ndarray): La matrice initiale
        precision (float, optional): La precision recquise pour considerer une valeur

    Returns:
        np.ndarray: La matrice triangle superieur obtenue apres elimination
    """
    (m, n) = A.shape  # Recuperation de la dimension de la matrice (lignes, colonnes)
    for k in range(min(m, n)):
        # Recherche du pivot
        p = trouveMaxi(A, k)
        maxi = abs(A[p, k])
        # Si pas de pivot trouve
        if maxi < precision:
            print('Le pivot est nul, la matrice est singulière')
            return  # Sortie du programme
        # Sinon
        # Echange des lignes k et p
        A[[k, p]] = A[[p, k]]
        # Elimination de gauss 
        for i in range(k+1, m):
            h = A[i, k] / A[k, k]
            for j in range(k, n):
                A[i, j] = A[i, j] - h * A[k, j]
                if abs(A[i, j]) < precision:
                    A[i, j] = 0
    return A

def retroSubstitution(A: np.ndarray, x: np.ndarray, b: np.ndarray):
    """ Algorithme de retro substitution

    Arguments:
        A (np.ndarray): Matrice apres elimination de gauss (triangle superieure)
        x (np.ndarray): Le vecteur de solution de forme [0]*len(b)
        b (np.ndarray): Le vecteur resultat du systeme lineaire
        
    Returns:
        np.ndarray: Le vecteur de solution
    """    
    n = b.size
    for i in range(n-1, -1, -1):
        s = 0
        # Calcul de la somme des produit scalaires des sous lignes de A et du vecteur x
        for j in range(i+1, n):
            s += A[i, j] * x[j]
        x[i] = (b[i] - s) / A[i, i]
        
    return x


def afficheSolution(x: np.ndarray):  # Affichage des résultats
    """ Affiches les resultats du programme

    Arguments:
        x (np.ndarray): Le vecteur resultat du systeme lineaire
    """
    # Affichage des intensite par resistance
    for i in range(x.size):
        print('i' + str(i+1) + ' = ' + str(round(1000 * x[i], 3)) + 'mA')
        # Gestion du sens
        if x[i] < 0:
            print(
                "L'intensité de courant ci-dessus est dans le sens inverse du schéma d'où le signe négatif.")
        elif x[i] > 0:
            print(
                "L'intensité de courant ci-dessus est dans le sens du schéma d'où le signe positif.")
        else:
            print(
                "L'intensité de courant ci-dessus est nulle")
    print('La résistance équivalente Req est égale à  : ' +
          str(round(V / (x[x.size-1] + x[x.size-2]), 1)/1000) + "kOhm") 
    print("Le courant d'intensité total fournit par le générateur est égal à :",
          np.round((x[x.size-1] + x[x.size-2])*1000, 3), "mA")


def inverseDoublement(A: np.ndarray, precision=0.000001):
    """ Inverse par doublement de tableau

    Arguments:
        A (np.ndarray): La matrice a inverser
        precision (float, optional): La precision recquise pour considerer une valeur 

    Returns:
        np.ndarray: La matrice inversee
    """    
    n = A.shape[1] # Recuperation du nombre de colonnes
    n2 = n*2 # Etendre A a n2 colonnes
    # Remplir la partie etendue par la matrice identite
    I = np.identity(n)
    A = np.hstack((A, I))
    for k in range(0, n):
        # Recherche du pivot
        p = trouveMaxi(A, k)
        maxi = abs(A[p, k])
        if maxi < precision:
            print('La matrice est non inversible')
            return
        # Echange des lignes k et p
        A[[k, p]] = A[[p, k]]
        h = A[k, k] # Le pivot
        # Calcul de l'inverse
        for j in range(k, n2):
            A[k, j] = A[k, j] / h
        for i in range(0, n):
            if i != k:
                h = A[i, k]
                for j in range(k, n2):
                    A[i, j] = A[i, j] - h * A[k, j]
    return A[:, n:]


A = inverseDoublement(R)

R = gaussOptiPartielle(R)

print('La matrice triangulaire supérieure est ci-dessous :')
print(np.round(R, -1))

# Calcul de la solution, problemes sous determines 1.6.2 x = tA(AtA)^-1*b

G = np.dot(R, A)
b = np.dot(G, b)

retroSubstitution(R, x, b)

afficheSolution(x)

print('Dans le cas où les résistances changent il faut modifier la valeur des résistances dans le programme R1,R2...,R10 idem pour la tension V')
