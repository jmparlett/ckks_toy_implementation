# CKKS Toy implementation for educational purposes

import numpy as np

M = 8 # Well use the roots of the M-th cyclotomic polynomial for encoding and decoding
N = M//2 # We are guaranteed the M-th cyclotomic is of degree M/2 if M is a power of 2

# XI is our complex root of unity, and want to work with the M-th
XI = np.exp(2 * np.pi * 1j / M)



from numpy.polynomial import Polynomial

class CKKSEncoder:
    '''
    Encodes complex vectors into polynomials
    '''

    def __init__(self,M: int) -> None:
        '''
        M: power of 2
        '''

        self.xi = np.exp(2 * np.pi * 1j / M)
        self.M = M

    @staticmethod
    def vandermond(xi : np.complex128, M: int) -> np.array:
        '''
        This construct the vandermond matrix where each row is the first N powers
        of one of the roots of the Mth cyclotomic polynomial. This is used in defining
        the mapping between spaces as a component of the linear system we solve to move
        between the two
        '''
        N = M//2

        matrix = []# initially a list but I believe the type annotation casts it to np.array

        for i in range(N):#0...N-1, working an N degree polynomial)
            # each row is powers of a different root
            root = xi ** (2 * i + 1) # add one to exclude 0 since 1 is never a root of our cyclotomic by definition
            row = []

            for j in range(N):
                row.append(root ** j)
            matrix.append(row)

        return matrix # its not explicit but guessing type annotation transforms this to np.array

    def encode(self, v: np.array) -> Polynomial:
        '''
        Encode a N dimensional vector to aPolynomial 
        '''
        # create vandermond matrix of roots
        A = CKKSEncoder.vandermond(self.xi,M)

        # compute coefficients that uniquely determine thePolynomial 
        coeffs = np.linalg.solve(A,v)

        p = Polynomial(coeffs) 
        return p

    def decode(self, p: Polynomial) -> np.array:
        '''
        Decode a polynomial to a plaintext vector, the original message
        by evaluating at the roots of the cyclotomic
        '''
        # create vandermond matrix of roots
        A = CKKSEncoder.vandermond(self.xi,M)

        return A @ p.coef