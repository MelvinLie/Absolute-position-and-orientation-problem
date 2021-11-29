import numpy as np
from scipy.special import sph_harm
import matplotlib.pyplot as plt
from scipy.special import lpmn
from scipy.special import factorial
from .solid_harmonics_base import *

'''
Compute the matrix to evaluate Br based on complex solid harmonic coefficients
'''
def compute_Br_mat_cmplx(r,theta,phi,L):
    M = len(r)

    N = (L+1)**2


    ret_mat = 1j*np.zeros((M,N))

    k = 0

    for l in range(L+1):

        for m in range(-l,l+1):

            ret_mat[:,k] = l*r**(l-1)*sph_harm(m,l,phi,theta)

            k += 1

    return ret_mat

'''
Compute the matrix to evaluate Btheta based on complex solid harmonic coefficients
'''
def compute_Bt_mat_cmplx(r,theta,phi,L):
    M = len(r)

    N = (L+1)**2


    ret_mat =  1j*np.zeros((M,N))

    k = 0

    for l in range(L+1):

        for m in range(-l,l+1):

            ret_mat[:,k] = r**(l-1)*dYdt(m,l,phi,theta)

            k += 1

    return ret_mat

'''
Compute the matrix to evaluate Bphi based on complex solid harmonic coefficients
'''
def compute_Bp_mat_cmplx(r,theta,phi,L):
    M = len(r)

    N = (L+1)**2


    ret_mat =  1j*np.zeros((M,N))

    k = 0

    for l in range(L+1):

        for m in range(-l,l+1):

            ret_mat[:,k] = 1j*m*r**(l-1)*sph_harm(m,l,phi,theta)/np.sin(theta)

            k += 1

    return ret_mat

'''
Compute the matrix to evaluate Bx based on complex solid harmonic coefficients
'''
def compute_Bx_mat_cmplx(p,L):
    M = p.shape[0]

    #spherical coordinates
    r = np.sqrt(p[:,0]**2+p[:,1]**2+p[:,2]**2)
    t = np.arccos(p[:,2]/r)
    p = np.arctan2(p[:,1],p[:,0])


    Br_mat = compute_Br_mat(r,t,p,L)
    Bt_mat = compute_Bt_mat(r,t,p,L)
    Bp_mat = compute_Bp_mat(r,t,p,L)

    Bx_mat = 1j*np.zeros(Br_mat.shape)

    for m in range(M):
        Bx_mat[m,:] = np.sin(t[m])*np.cos(p[m])*Br_mat[m,:] \
                     + np.cos(t[m])*np.cos(p[m])*Bt_mat[m,:] \
                     - np.sin(p[m])*Bp_mat[m,:]


    return Bx_mat

'''
Compute the matrix to evaluate By based on complex solid harmonic coefficients
'''
def compute_By_mat_cmplx(p,L):
    M = p.shape[0]

    #spherical coordinates
    r = np.sqrt(p[:,0]**2+p[:,1]**2+p[:,2]**2)
    t = np.arccos(p[:,2]/r)
    p = np.arctan2(p[:,1],p[:,0])


    Br_mat = compute_Br_mat(r,t,p,L)
    Bt_mat = compute_Bt_mat(r,t,p,L)
    Bp_mat = compute_Bp_mat(r,t,p,L)

    By_mat = 1j*np.zeros(Br_mat.shape)

    for m in range(M):

        By_mat[m,:] = np.sin(t[m])*np.sin(p[m])*Br_mat[m,:] \
                     + np.cos(t[m])*np.sin(p[m])*Bt_mat[m,:] \
                     + np.cos(p[m])*Bp_mat[m,:]

        #Bz_mat[m,:] = np.cos(t)*Br_mat[m,:] \
        #             - np.sin(t)*Bt_mat[m,:]

    return By_mat

'''
Compute the matrix to evaluate Bz based on complex solid harmonic coefficients
'''
def compute_Bz_mat_cmplx(p,L):
    M = p.shape[0]

    #spherical coordinates
    r = np.sqrt(p[:,0]**2+p[:,1]**2+p[:,2]**2)
    t = np.arccos(p[:,2]/r)
    p = np.arctan2(p[:,1],p[:,0])

    Br_mat = compute_Br_mat(r,t,p,L)
    Bt_mat = compute_Bt_mat(r,t,p,L)

    Bz_mat = 1j*np.zeros(Br_mat.shape)

    for m in range(M):

        Bz_mat[m,:] = np.cos(t[m])*Br_mat[m,:] \
                     - np.sin(t[m])*Bt_mat[m,:]

    return Bz_mat