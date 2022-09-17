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
