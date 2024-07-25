import numpy as np
from scipy import integrate

class cosmology():
    def __init__(self, H0=70, omega_m=0.3, omega_l=0.7):
        '''
        Class to facilitate cosmological computations in a
        universe with a critical density.

        Parameters:
        -----------
            H0: `int` or `float`
                Hubble parameter at z = 0
            omega_m: `int` or `float`
                Density parameter of matter
            omega_l: `int` or `float`
                Density parameter of dark energy
        '''
        self._H0 = H0 # hubble constant at z=0
        self._omega_m = omega_m # density parameter of matter
        self._omega_l = omega_l # density parameter of dark energy
        # density parameter equivalent from curvature (omega = 1)
        self._omega_k = 1 - omega_m - omega_l 
        self._c = 3e5 # speed of light in km/s
        self._D_H = self._c/H0 # hubble distance
    
    def _H(self, z):
        '''
        Compute the Hubble parameter at a given redshift.

        Inputs:
        -------
        z: `int`, `float`, or `numpy array`
            Redshift at which to compute the Hubble parameter

        Outputs:
        --------
        H: `float`
            Hubble parameter at redshift z (km/s / Mpc)
        '''

        # Friedmann equation in useful form from lecture notes
        H = self._H0 * np.sqrt( self._omega_m*(1+z)**3 + \
                              self._omega_k*(1+z)**2 + self._omega_l )
        return H

    def _D_C(self, z):
        '''
        Compute the comoving distance at a given redshift.

        Inputs:
        -------
        z: `int`, `float`, or `numpy array`
            Redshift at which to compute the Hubble parameter

        Outputs:
        --------
        D_C: `float`
            Comoving distance at redshift z (Mpc)
        '''

        # define c/H because that's the argument of the integral
        H_inv = lambda z: self._c / self._H(z)

        # if z is an int or float, just calculate r at that value
        if isinstance(z, (int, float)):
            D_C, _ = integrate.quad(H_inv, a=0, b=z)
            return D_C

        # otherwise assume z is an array, then compute r at each z value
        D_C = np.zeros_like(z)
        for i in range(len(z)):
            D_C[i], _ = integrate.quad(H_inv, a=0, b=z[i]) 
        return D_C

    def _D_M(self, z):
        '''
        Compute the transverse comoving distance (proper distance) at a
        given redshift as a function of cosmology.

        Inputs:
        -------
        z: `int`, `float`, or `numpy array`
            Redshift at which to compute the Hubble parameter

        Outputs:
        --------
        D_M: `float`
            Comoving distance at redshift z (Mpc)
        '''

        # if the universe if flat, D_M == comoving distance
        if self._omega_k == 0:
            D_M = self._D_C(z)
            return D_M

        # if not flat, use the analytic solution (Hogg 2000)
        if self._omega_k > 0:
            D_M = self._D_H * 1/np.sqrt(self._omega_k) * \
                  np.sinh(np.sqrt(self._omega_k)*self._D_C(z)/self._D_H)
        elif self._omega_k < 0:
            D_M = self._D_H * 1/np.sqrt(self._omega_k) * \
                  np.sin(np.sqrt(self._omega_k)*self._D_C(z)/self._D_H)
        return D_M


    def _D_A(self, z):
        '''
        Compute the angular diameter distance at a given redshift.

        Inputs:
        -------
        z: `int`, `float`, or `numpy array`
            Redshift at which to compute the Hubble parameter

        Outputs:
        --------
        D_A: `float`
            Angular diameter distance at redshift z (Mpc)
        '''
        D_A = self._D_M(z) / (1+z)
        return D_A

    def _D_L(self, z):
        '''
        Compute the luminosity distance at a given redshift.

        Inputs:
        -------
        z: `int`, `float`, or `numpy array`
            Redshift at which to compute the Hubble parameter

        Outputs:
        --------
        D_A: `float`
            Luminosity distance at redshift z (Mpc)
        '''
        D_L = self._D_M(z) * (1+z)
        return D_L

    def _distmod(self, z):
        '''
        Compute the distance modulus at a given redshift.

        Inputs:
        -------
        z: `int`, `float`, or `numpy array`
            Redshift at which to compute the Hubble parameter

        Outputs:
        --------
        distmod: `float`
            Distance modulus at redshift z
        '''
        D_L = self._D_L(z) # luminosity distance
        distmod = 5*np.log10(D_L*1e6/10)
        return distmod

    def _volume(self, z):
        '''
        Compute the differential comoving volume at a given redshift.

        Inputs:
        -------
        z: `int`, `float`, or `numpy array`
            Redshift at which to compute the Hubble parameter

        Outputs:
        --------
        V: `float`
             Differential volume per solid angle per unit 
             redshift (Mpc^3 / str)
        '''
        V = self._c * self._D_M(z)**2 / self._H(z)
        return V

    def _t(self, z):
        '''
        Compute the age of the universe at a given redshift.

        Inputs:
        -------
        z: `int`, `float`, or `numpy array`
            Redshift at which to compute the Hubble parameter

        Outputs:
        --------
        t: `float`
             Age of the universe at redshift z (Myr)
        '''
        # define the argument of the integral
        H_inv = lambda z: 1 / self._H(z) / (1+z)

        # if z is an int or float, just calculate t at that value
        if isinstance(z, (int, float)):
            t, _ = integrate.quad(H_inv, a=0, b=z)
            return t * 3.09e19 / 31500000 * 1e-6 # t in Myr

        # otherwise assume z is an array, then compute t at each z value
        t = np.zeros_like(z)
        for i in range(len(z)):
            t[i], _ = integrate.quad(H_inv, a=0, b=z[i])

        return t * 3.09e19 / 31500000 * 1e-6 # t in Myr