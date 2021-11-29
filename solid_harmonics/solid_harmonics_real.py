import numpy as np
from scipy.special import sph_harm
import matplotlib.pyplot as plt
from scipy.special import lpmn
from scipy.special import factorial
from .solid_harmonics_base import *

'''
Compute the matrix to evaluate Br based on real solid harmonic coefficients
'''
def compute_Br_mat_real(r,theta,phi,L):

    M = len(r)

    N = (L+1)**2


    ret_mat = np.zeros((M,N))

    k = 0

    for l in range(L+1):

        ret_mat[:,k] = l*r**(l-1)*np.real(sph_harm(0,l,phi,theta))
        k += 1

        for m in range(1,l+1):

            tmp = 2*l*r**(l-1)*sph_harm(m,l,phi,theta)

            ret_mat[:,k] = np.real(tmp)
            k += 1

            ret_mat[:,k] = -np.imag(tmp)
            k += 1

    return ret_mat

'''
Compute the matrix to evaluate Bt based on real solid harmonic coefficients
'''
def compute_Bt_mat_real(r,theta,phi,L):

    M = len(r)

    N = (L+1)**2

    ret_mat = np.zeros((M,N))

    k = 0

    for l in range(L+1):

        ret_mat[:,k] = r**(l-1)*np.real(dYdt(0,l,phi,theta))
        k += 1

        for m in range(1,l+1):

            tmp = 2*r**(l-1)*dYdt(m,l,phi,theta)

            ret_mat[:,k] = np.real(tmp)
            k += 1

            ret_mat[:,k] = -np.imag(tmp)
            k += 1

    return ret_mat

'''
Compute the matrix to evaluate Bp based on real solid harmonic coefficients
'''
def compute_Bp_mat_real(r,theta,phi,L):

    M = len(r)

    N = (L+1)**2

    ret_mat = np.zeros((M,N))

    k = 0

    for l in range(L+1):

        k += 1

        for m in range(1,l+1):

            tmp = 2j*m*r**(l-1)*sph_harm(m,l,phi,theta)/np.sin(theta)

            ret_mat[:,k] = np.real(tmp)
            k += 1

            ret_mat[:,k] = -np.imag(tmp)
            k += 1

    return ret_mat


'''
Compute the matrix to evaluate Bx based on real solid harmonic coefficients
'''
def compute_Bx_mat_real(p,L):
    M = p.shape[0]

    #spherical coordinates
    r = np.sqrt(p[:,0]**2+p[:,1]**2+p[:,2]**2)
    t = np.arccos(p[:,2]/r)
    p = np.arctan2(p[:,1],p[:,0])


    Br_mat = compute_Br_mat_real(r,t,p,L)
    Bt_mat = compute_Bt_mat_real(r,t,p,L)
    Bp_mat = compute_Bp_mat_real(r,t,p,L)

    Bx_mat = 1j*np.zeros(Br_mat.shape)

    for m in range(M):
        Bx_mat[m,:] = np.sin(t[m])*np.cos(p[m])*Br_mat[m,:] \
                     + np.cos(t[m])*np.cos(p[m])*Bt_mat[m,:] \
                     - np.sin(p[m])*Bp_mat[m,:]


    return Bx_mat

'''
Compute the matrix to evaluate By based on real solid harmonic coefficients
'''
def compute_By_mat_real(p,L):
    M = p.shape[0]

    #spherical coordinates
    r = np.sqrt(p[:,0]**2+p[:,1]**2+p[:,2]**2)
    t = np.arccos(p[:,2]/r)
    p = np.arctan2(p[:,1],p[:,0])


    Br_mat = compute_Br_mat_real(r,t,p,L)
    Bt_mat = compute_Bt_mat_real(r,t,p,L)
    Bp_mat = compute_Bp_mat_real(r,t,p,L)

    By_mat = 1j*np.zeros(Br_mat.shape)

    for m in range(M):

        By_mat[m,:] = np.sin(t[m])*np.sin(p[m])*Br_mat[m,:] \
                     + np.cos(t[m])*np.sin(p[m])*Bt_mat[m,:] \
                     + np.cos(p[m])*Bp_mat[m,:]

        #Bz_mat[m,:] = np.cos(t)*Br_mat[m,:] \
        #             - np.sin(t)*Bt_mat[m,:]

    return By_mat

'''
Compute the matrix to evaluate Bz based on real solid harmonic coefficients
'''
def compute_Bz_mat_real(p,L):
    M = p.shape[0]

    #spherical coordinates
    r = np.sqrt(p[:,0]**2+p[:,1]**2+p[:,2]**2)
    t = np.arccos(p[:,2]/r)
    p = np.arctan2(p[:,1],p[:,0])

    Br_mat = compute_Br_mat_real(r,t,p,L)
    Bt_mat = compute_Bt_mat_real(r,t,p,L)

    Bz_mat = 1j*np.zeros(Br_mat.shape)

    for m in range(M):

        Bz_mat[m,:] = np.cos(t[m])*Br_mat[m,:] \
                     - np.sin(t[m])*Bt_mat[m,:]

    return Bz_mat

'''
Compute the matrix to evaluate Hall probe measurements, based on real solid
harmonic coefficients.
'''
def compute_meas_mat_real(p,s,n,L):
    M = p.shape[0]

    #spherical coordinates
    r = np.sqrt(p[:,0]**2+p[:,1]**2+p[:,2]**2)
    t = np.arccos(p[:,2]/r)
    p = np.arctan2(p[:,1],p[:,0])


    Br_mat = compute_Br_mat_real(r,t,p,L)
    Bt_mat = compute_Bt_mat_real(r,t,p,L)

    Bx_mat = np.zeros(Br_mat.shape)
    By_mat = np.zeros(Br_mat.shape)
    Bz_mat = np.zeros(Br_mat.shape)

    for m in range(M):
        Bx_mat[m,:] = np.sin(t[m])*np.cos(p[m])*Br_mat[m,:] \
                     + np.cos(t[m])*np.cos(p[m])*Bt_mat[m,:]

        By_mat[m,:] = np.sin(t[m])*np.sin(p[m])*Br_mat[m,:] \
                     + np.cos(t[m])*np.sin(p[m])*Bt_mat[m,:]

        Bz_mat[m,:] = np.cos(t[m])*Br_mat[m,:] \
                     - np.sin(t[m])*Bt_mat[m,:]

    return s*(n[0]*Bx_mat + n[1]*By_mat + n[2]*Bz_mat)


'''
Evaluate Hall probe measurements, based on real solid  harmonic coefficients.
'''
def compute_meas_real(p,s,n,c):
    #number of measurements
    M = p.shape[0]
    #number of Dofs for field
    L = np.int32(np.sqrt(len(c)) - 1)
    #spherical coordinates
    r = np.sqrt(p[:,0]**2+p[:,1]**2+p[:,2]**2)
    t = np.arccos(p[:,2]/r)
    p = np.arctan2(p[:,1],p[:,0])


    Br_mat = compute_Br_mat_real(r,t,p,L)
    Bt_mat = compute_Bt_mat_real(r,t,p,L)

    Bx_mat = np.zeros(Br_mat.shape)
    By_mat = np.zeros(Br_mat.shape)
    Bz_mat = np.zeros(Br_mat.shape)

    for m in range(M):
        Bx_mat[m,:] = np.sin(t[m])*np.cos(p[m])*Br_mat[m,:] \
                     + np.cos(t[m])*np.cos(p[m])*Bt_mat[m,:]

        By_mat[m,:] = np.sin(t[m])*np.sin(p[m])*Br_mat[m,:] \
                     + np.cos(t[m])*np.sin(p[m])*Bt_mat[m,:]

        Bz_mat[m,:] = np.cos(t[m])*Br_mat[m,:] \
                     - np.sin(t[m])*Bt_mat[m,:]

    return s*(n[0]*Bx_mat + n[1]*By_mat + n[2]*Bz_mat) @ c

'''
Evaluate Hall probe measurements, based on real solid  harmonic coefficients,
given position and orientation of the sensor.
'''
def M_real(r,c,ay,az,bx,bz,gx,gy,dx,dy,dz,sx,sy,sz):
    
    #number of measurements
    M = r.shape[0]

    #output vector
    y = np.zeros((3*M,))

    #probe x:
    #orientation vector
    nx = np.array([np.cos(gx)*np.cos(bx),
                   np.sin(gx)*np.cos(bx),
                  -np.sin(bx)])
    #probe position
    rx = r.copy()
    rx[:,0] += dx[0]
    rx[:,1] += dx[1]
    rx[:,2] += dx[2]

    #compute measurements
    y[:M] = compute_meas_real(rx,sx[0],nx,c) + sx[1]

    #probe y:
    #orientation vector
    ny = np.array([-np.sin(gy)*np.cos(ay),
                   np.cos(gy)*np.cos(ay),
                   np.sin(ay)])
    #probe position
    ry = r.copy()
    ry[:,0] += dy[0]
    ry[:,1] += dy[1]
    ry[:,2] += dy[2]

    #compute measurements
    y[M:2*M] = compute_meas_real(ry,sy[0],ny,c) + sy[1]

    #probe z:
    #orientation vector
    nz = np.array([np.sin(bz)*np.cos(az),
                   -np.sin(az),
                   np.cos(bz)*np.cos(az)])

    #probe position
    rz = r.copy()
    rz[:,0] += dz[0]
    rz[:,1] += dz[1]
    rz[:,2] += dz[2]

    #compute measurements
    y[2*M:] = compute_meas_real(rz,sz[0],nz,c) + sz[1]

    return y


'''
Compute the Jacobian of M given above, for all parameters:
Example: (K-field unknowns)
[ K-cols ] , [          6 cols       ] , [    9 cols     ] , [   3 cols   ]
     c     ,   ay, az, bx, bz, gx , gy ,     dx, dy, dz    ,   sx, sy, az
If only some of these parameters are unknown, extract the corresponding ones
from the large Jacobian:
Example: only angles:
J = Jac_M(...)[:,:K+6]
'''
def Jac_M_real(r,c,ay,az,bx,bz,gx,gy,dx,dy,dz,sx,sy,sz):

    #stepsize for the approximation in dx dy dz
    eps_d = 1e-6

    #number of measurements
    num_meas = r.shape[0]


    L = np.int32(np.sqrt(len(c)) - 1)
    #number of field DoFs
    N = (L+1)**2

    #output Matrix
    J = np.zeros((3*num_meas,N +18))

    #***************************************************
    #Compute B field matrices
    #***************************************************
    #probe x:
    #orientation vector
    nx = np.array([np.cos(gx)*np.cos(bx),
                   np.sin(gx)*np.cos(bx),
                  -np.sin(bx)])
    #probe position
    rx = r.copy()
    rx[:,0] += dx[0]
    rx[:,1] += dx[1]
    rx[:,2] += dx[2]

    #spherical coordinates
    Rx = np.sqrt(rx[:,0]**2+rx[:,1]**2+rx[:,2]**2)
    Tx = np.arccos(rx[:,2]/Rx)
    Px = np.arctan2(rx[:,1],rx[:,0])

    Br_mat_x = compute_Br_mat_real(Rx,Tx,Px,L)
    Bt_mat_x = compute_Bt_mat_real(Rx,Tx,Px,L)

    Bx_mat_x = np.zeros(Br_mat_x.shape)
    Bx_mat_y = np.zeros(Br_mat_x.shape)
    Bx_mat_z = np.zeros(Br_mat_x.shape)


    for m in range(num_meas):
        Bx_mat_x[m,:] = np.sin(Tx[m])*np.cos(Px[m])*Br_mat_x[m,:] \
                     + np.cos(Tx[m])*np.cos(Px[m])*Bt_mat_x[m,:]

        Bx_mat_y[m,:] = np.sin(Tx[m])*np.sin(Px[m])*Br_mat_x[m,:] \
                     + np.cos(Tx[m])*np.sin(Px[m])*Bt_mat_x[m,:]

        Bx_mat_z[m,:] = np.cos(Tx[m])*Br_mat_x[m,:] \
                     - np.sin(Tx[m])*Bt_mat_x[m,:]

    #probe y:
    #orientation vector
    ny = np.array([-np.sin(gy)*np.cos(ay),
                   np.cos(gy)*np.cos(ay),
                   np.sin(ay)])
    #probe position
    ry = r.copy()
    ry[:,0] += dy[0]
    ry[:,1] += dy[1]
    ry[:,2] += dy[2]

    #spherical coordinates
    Ry = np.sqrt(ry[:,0]**2+ry[:,1]**2+ry[:,2]**2)
    Ty = np.arccos(ry[:,2]/Ry)
    Py = np.arctan2(ry[:,1],ry[:,0])

    Br_mat_y = compute_Br_mat_real(Ry,Ty,Py,L)
    Bt_mat_y = compute_Bt_mat_real(Ry,Ty,Py,L)

    By_mat_x = np.zeros(Br_mat_y.shape)
    By_mat_y = np.zeros(Br_mat_y.shape)
    By_mat_z = np.zeros(Br_mat_y.shape)

    for m in range(num_meas):
        By_mat_x[m,:] = np.sin(Ty[m])*np.cos(Py[m])*Br_mat_y[m,:] \
                     + np.cos(Ty[m])*np.cos(Py[m])*Bt_mat_y[m,:]

        By_mat_y[m,:] = np.sin(Ty[m])*np.sin(Py[m])*Br_mat_y[m,:] \
                     + np.cos(Ty[m])*np.sin(Py[m])*Bt_mat_y[m,:]

        By_mat_z[m,:] = np.cos(Ty[m])*Br_mat_y[m,:] \
                     - np.sin(Ty[m])*Bt_mat_y[m,:]

    #probe z:
    #orientation vector
    nz = np.array([np.sin(bz)*np.cos(az),
                   -np.sin(az),
                   np.cos(bz)*np.cos(az)])

    #probe position
    rz = r.copy()
    rz[:,0] += dz[0]
    rz[:,1] += dz[1]
    rz[:,2] += dz[2]

    #spherical coordinates
    Rz = np.sqrt(rz[:,0]**2+rz[:,1]**2+rz[:,2]**2)
    Tz = np.arccos(rz[:,2]/Rz)
    Pz = np.arctan2(rz[:,1],rz[:,0])

    Br_mat_z = compute_Br_mat_real(Rz,Tz,Pz,L)
    Bt_mat_z = compute_Bt_mat_real(Rz,Tz,Pz,L)

    Bz_mat_x = np.zeros(Br_mat_z.shape)
    Bz_mat_y = np.zeros(Br_mat_z.shape)
    Bz_mat_z = np.zeros(Br_mat_z.shape)

    for m in range(num_meas):
        Bz_mat_x[m,:] = np.sin(Tz[m])*np.cos(Pz[m])*Br_mat_z[m,:] \
                     + np.cos(Tz[m])*np.cos(Pz[m])*Bt_mat_z[m,:]

        Bz_mat_y[m,:] = np.sin(Tz[m])*np.sin(Pz[m])*Br_mat_z[m,:] \
                     + np.cos(Tz[m])*np.sin(Pz[m])*Bt_mat_z[m,:]

        Bz_mat_z[m,:] = np.cos(Tz[m])*Br_mat_z[m,:] \
                     - np.sin(Tz[m])*Bt_mat_z[m,:]

    #***************************************************
    #n derivatives
    #***************************************************
    nx_dgx = np.array([-np.sin(gx)*np.cos(bx),
                   np.cos(gx)*np.cos(bx),
                   0.])

    nx_dbx = np.array([-np.cos(gx)*np.sin(bx),
                   -np.sin(gx)*np.sin(bx),
                  -np.cos(bx)])


    ny_day = np.array([np.sin(gy)*np.sin(ay),
                   -np.cos(gy)*np.sin(ay),
                   np.cos(ay)])

    ny_dgy = np.array([-np.cos(gy)*np.cos(ay),
                   -np.sin(gy)*np.cos(ay),
                   0.])

    nz_dbz = np.array([np.cos(bz)*np.cos(az),
                   0.,
                   -np.sin(bz)*np.cos(az)])

    nz_daz = np.array([-np.sin(bz)*np.sin(az),
                   -np.cos(az),
                   -np.cos(bz)*np.sin(az)])



    #***************************************************
    #execute mat vec products
    #***************************************************

    Bx_x = Bx_mat_x @ c
    Bx_y = Bx_mat_y @ c
    Bx_z = Bx_mat_z @ c

    By_x = By_mat_x @ c
    By_y = By_mat_y @ c
    By_z = By_mat_z @ c

    Bz_x = Bz_mat_x @ c
    Bz_y = Bz_mat_y @ c
    Bz_z = Bz_mat_z @ c

    #***************************************************
    #Assemble Jacobian
    #***************************************************
    #probe x:
    J[:num_meas,:N] = sx[0]*(nx[0]*Bx_mat_x + nx[1]*Bx_mat_y + nx[2]*Bx_mat_z)

    #probe y:
    J[num_meas:2*num_meas,:N] = sy[0]*(ny[0]*By_mat_x + ny[1]*By_mat_y + ny[2]*By_mat_z)

    #probe z:
    J[2*num_meas:,:N] = sz[0]*(nz[0]*Bz_mat_x + nz[1]*Bz_mat_y + nz[2]*Bz_mat_z)

    #alpha_y
    J[num_meas:2*num_meas,N] = sy[0]*(ny_day[0]*By_x \
                   + ny_day[1]*By_y \
                   + ny_day[2]*By_z )

    #alpha_z
    J[2*num_meas:,N+1] = sz[0]*(nz_daz[0]*Bz_x \
                   + nz_daz[1]*Bz_y \
                   + nz_daz[2]*Bz_z )

    #beta_x
    J[:num_meas,N+2]    = sx[0]*(nx_dbx[0]*Bx_x \
                   + nx_dbx[1]*Bx_y \
                   + nx_dbx[2]*Bx_z )

    #beta_z
    J[2*num_meas:,N+3]  = sz[0]*(nz_dbz[0]*Bz_x \
                   + nz_dbz[2]*Bz_z )

    #gamma x
    J[:num_meas,N+4]    = sx[0]*(nx_dgx[0]*Bx_x \
                   + nx_dgx[1]*Bx_y )

    #gamma y
    J[num_meas:2*num_meas,N+5] = sy[0]*(ny_dgy[0]*By_x \
                   + ny_dgy[1]*By_y )


    #for numerical approximation
    y_0 = M_real(r,c,ay,az,bx,bz,gx,gy,dx,dy,dz,sx,sy,sz)

    #d_x^x
    tmp_d = dx.copy()
    tmp_d[0] += eps_d
    J[:,N+6] = (M_real(r,c,ay,az,bx,bz,gx,gy,tmp_d,dy,dz,sx,sy,sz) - y_0) / eps_d

    #d_x^y
    tmp_d = dx.copy()
    tmp_d[1] += eps_d
    J[:,N+7] = (M_real(r,c,ay,az,bx,bz,gx,gy,tmp_d,dy,dz,sx,sy,sz) - y_0) / eps_d

    #d_x^z
    tmp_d = dx.copy()
    tmp_d[2] += eps_d
    J[:,N+8] = (M_real(r,c,ay,az,bx,bz,gx,gy,tmp_d,dy,dz,sx,sy,sz) - y_0) / eps_d

    #d_y^x
    tmp_d = dy.copy()
    tmp_d[0] += eps_d
    J[:,N+9] = (M_real(r,c,ay,az,bx,bz,gx,gy,dx,tmp_d,dz,sx,sy,sz) - y_0) / eps_d

    #d_y^y
    tmp_d = dy.copy()
    tmp_d[1] += eps_d
    J[:,N+10] = (M_real(r,c,ay,az,bx,bz,gx,gy,dx,tmp_d,dz,sx,sy,sz) - y_0) / eps_d

    #d_y^z
    tmp_d = dy.copy()
    tmp_d[2] += eps_d
    J[:,N+11] = (M_real(r,c,ay,az,bx,bz,gx,gy,dx,tmp_d,dz,sx,sy,sz) - y_0) / eps_d

    #d_z^x
    tmp_d = dz.copy()
    tmp_d[0] += eps_d
    J[:,N+12] = (M_real(r,c,ay,az,bx,bz,gx,gy,dx,dy,tmp_d,sx,sy,sz) - y_0) / eps_d

    #d_z^y
    tmp_d = dz.copy()
    tmp_d[1] += eps_d
    J[:,N+13] = (M_real(r,c,ay,az,bx,bz,gx,gy,dx,dy,tmp_d,sx,sy,sz) - y_0) / eps_d

    #d_z^z
    tmp_d = dz.copy()
    tmp_d[2] += eps_d
    J[:,N+14] = (M_real(r,c,ay,az,bx,bz,gx,gy,dx,dy,tmp_d,sx,sy,sz) - y_0) / eps_d

    #Sensitivities:
    J[:num_meas,N+15]           = (nx[0]*Bx_x + nx[1]*Bx_y + nx[2]*Bx_z)
    J[num_meas:2*num_meas,N+16] = (ny[0]*By_x + ny[1]*By_y + ny[2]*By_z)
    J[2*num_meas:,N+17]         = (nz[0]*Bz_x + nz[1]*Bz_y + nz[2]*Bz_z)

    return -1*J
