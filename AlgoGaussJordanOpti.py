
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
