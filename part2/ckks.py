# CKKS Toy implementation for educational purposes

import numpy as np

# M = 8 # Well use the roots of the M-th cyclotomic polynomial for encoding and decoding
# N = M//2 # We are guaranteed the M-th cyclotomic is of degree M/2 if M is a power of 2

# XI is our complex root of unity, and want to work with the M-th
# XI = np.exp(2 * np.pi * 1j / M)



from numpy.polynomial import Polynomial

class CKKSEncoder:
    '''
    Encodes complex vectors into polynomials
    '''

    def __init__(self,M: int, scale:float = 64) -> None:
        '''
        M: power of 2
        '''

        self.xi = np.exp(2 * np.pi * 1j / M)
        self.M = M
        self.create_sigma_R_basis()
        self.scale = scale

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

    def sigma_inverse(self, v: np.array) -> Polynomial:
        '''
        Encode a N dimensional vector to aPolynomial 
        '''
        # create vandermond matrix of roots
        A = CKKSEncoder.vandermond(self.xi,self.M)

        # compute coefficients that uniquely determine thePolynomial 
        coeffs = np.linalg.solve(A,v)

        p = Polynomial(coeffs) 
        return p

    def sigma(self, p: Polynomial) -> np.array:
        '''
        Decode a polynomial to a plaintext vector, the original message
        by evaluating at the roots of the cyclotomic
        '''
        # create vandermond matrix of roots
        A = CKKSEncoder.vandermond(self.xi,self.M)

        return A @ p.coef

    def encode(self, z : np.array):
        '''
        Encode takes a vector to a polynomial
        '''
        pi_z = self.pi_inverse(z) # expand from C^{N/2} to H
        scaled_pi_z = self.scale * pi_z # scale to maintain precision
        rounded_scale_pi_z = self.sigma_R_discretization(scaled_pi_z) 
        p = self.sigma_inverse(rounded_scale_pi_z)

        # a lot of this is sketch to me
        coef = np.round(np.real(p.coef)).astype(int) # seems a lil cheatsy
        p = Polynomial(coef)
        return p

    def decode(self, p:Polynomial):
        '''
        Takes a polynomial back to a vector
        '''
        rescaled_p = p / self.scale # remove scale
        z = self.sigma(rescaled_p) # compute vector from poly
        pi_z = self.pi(z) # cut vec in half to go from H to C^{N/2}
        return pi_z




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

    def sigma_R_discretization(self,z : np.array) -> np.array:
        '''compute closest lattice vector to z in sigma(R) using coordinate wise random rounding algorithm'''
        coordinates = self.compute_basis_coordinates(z) # project into sigma(R) lattice basis

        rounded_coordinates = coordinate_wise_random_rounding(coordinates)

        y = np.matmul(self.sigma_R_basis.T, rounded_coordinates) # I have no idea what this is for
        # It might be performing change of basis here?

        return y




# coordinate wise random rounding utils
def round_remainder(coordinates):
    '''
    gives vector of distances from each coordinate to its floor
    '''
    return coordinates - np.floor(coordinates)

def coordinate_wise_random_rounding(coordinates):
    '''
    This is the algorithm discussed in the post. It rounds each coordinate to its floor
    or ceiling with probability dependent on distance from of each. It is most likely
    rounded the one it has min distance to.
    '''
    r = round_remainder(coordinates)

    #c is distance to floor, c-1 is distance to ceiling (these are not absolute values, c will be pos, c-1 will be neg). 
    # If c is small then 1-c is large and we wil pick c with probability
    # similarly for c-1
    f = np.array([np.random.choice([c,c-1], 1, p = [1-c,c]) for c in r]).reshape(-1)

    # because c/c-1 are not absolute this ensure we either round up for down by subtraction distance to floor or adding distance to ceil
    round_coordinates = coordinates - f

    # convert from float to int
    rounded_coordinates = [int(coeff) for coeff in round_coordinates]
    return round_coordinates