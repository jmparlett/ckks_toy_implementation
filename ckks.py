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
        self.create_sigma_R_basis()

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


    # Part 2 additions.

    def pi(self, z: np.array) -> np.array:
        '''
        Project a vector in H to C^{N/2} by discarding the second half of the vector.
        '''

        N = self.M // 4 # recall N = M/2 so were going from C^N = C^M/2 to C^N/2 = C^M/4
        return z[:N]

    def pi_inverse(self, z: np.array) -> np.array:
        '''
        Expand vector to C^{N/2} to H by doubling its size and copying the complex conjugates of its coordinates
        to the 2nd half of the vector
        '''

        # reverse z since we need z_i = conjugate(z_{N-i})
        z_conjugate = z[::-1]

        # compute conjugate of each term
        z_conjugate = [np.conjugate(x) for x in z_conjugate]

        return np.concatenate([z,z_conjugate])

    def create_sigma_R_basis(self):
        '''Create the orthogonal basis for sigma(R) from the orthogonal basis for R'''
        self.sigma_R_basis = np.array(self.vandermond(self.xi, self.M)).T

    def compute_basis_coordinates(self, z : np.array) -> np.array:
        '''
        Compute the projection of z in H to sigma(R), after this is performed we want to
        discretize the result to get the closest lattice vector
        '''
        # compute the sum of the projections of z onto each of the basis vectors
        return np.array([np.real(np.vdot(z,b) / np.vdot(b,b)) for b in self.sigma_R_basis])