'''Compute the finite worm-like chain (FWLC) approximation to arbitrary precision, as discussed in:
Yeonee Seol, Jinyu Li, Philip C. Nelson, Thomas T. Perkins, M.D. Betterton,
Elasticity of Short DNA Molecules: Theory and Experiment for Contour Lengths of 0.6–7μm,
Biophysical Journal, Volume 93, Issue 12, 2007, Pages 4360-4373, https://doi.org/10.1529/biophysj.107.112995.
'''

import numpy as np

from scipy.sparse import diags
from scipy.sparse.linalg import eigs
from scipy.linalg import expm
from scipy.special import sph_harm
from scipy.integrate import quad
from scipy.special import jn

class FWLC():

    def __init__(self, N=30):
        '''construct a FWLC

        Args:
            N (int): highest order of spherical harmonics to include in calculations. Series will be truncated at l = N.
                    Note that m = 0 for all harmonics due to azimuthal symmetry.
        '''

        #initialize H and A
        self._compute_H_A(0, N)

    def _compute_H_A(self, f, N=30):
        '''compute the matrices H and A, see Seol et al eq 11, 12, 17
        
        Args:
            f (float): normalized force, f = lp * F / (kB T)
            N (int): highest order of spherical harmonics to include in calculations. Series will be truncated at l = N.
                    Note that m = 0 for all harmonics due to azimuthal symmetry.
        '''

        diag = [-j * (j+1)/2 for j in np.arange(N)]
        offdiag = [ f *(j+1)/np.sqrt((2*j+1)*(2*j+3)) for j in np.arange(N-1)]

        
        self.H = diags([offdiag, diag, offdiag], [-1,0,1], format='csc')
        
        A = expm(self.H)
        vals, _ = eigs(A.toarray())
        maxval = np.abs(np.max(vals))
        self.A = A/maxval

    def setBoundaryHalfConstrained(self, N=30):
        psi_0 = np.array([quad(lambda x: self.Yj0(j, x), 0,1)[0] for j in np.arange(N)])
        psi_0 = psi_0/psi_0[0] #set first element to 1 by normalizing in an arbitrary way. This does not matter since we will be computing derivatives of the log of Z


    def Yj0(self, j, cosphi):
        '''evaluate spherical harmonics for m = 0'''
        return np.real(sph_harm(0, j, 0, np.arccos(cosphi)) * 2 * np.pi)

    def g(self, cosgamma, f, r):
        '''eq 24 in Seol et al'''
        integrand = lambda x: np.exp(-f*r*(x*cosgamma+1)) *jn(0,- np.complex(0,f*r*np.sin(np.arccos(cosgamma))*np.sin(np.arccos(x))))
        result = 2 * np.pi *np.real(quad(integrand, -1, 0))
        return result