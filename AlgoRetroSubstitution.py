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
