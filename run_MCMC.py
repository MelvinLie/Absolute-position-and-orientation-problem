import numpy as np
#import my_tools as my
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import gamma as gamma_distr
from scipy.optimize import least_squares
import pandas as pd
from scipy.linalg import lstsq
import numpy.matlib
import time
import scipy as sci
import io_tools as iot
import solid_harmonics as sh

enable_MCMC = True

'''*****************************************************************************
Preface
We are working in mm here!
We do so, to avoid high variances in the prior for the field coefficients!
The coefficients of the solid harmonics scale with r^l. If we are working in m,
these coefficients will get very large for high l. We therefore choose to work
in mm.
*****************************************************************************'''

'''*****************************************************************************
(1) Load Measurement Data
*****************************************************************************'''
meas_filename = 'meas_data/ELENA/august_27/cone_map.txt'

r,U,r0 = iot.read_measurement_data(meas_filename,center = np.array([1e-3,0.,2e-3]),filter = 3)

num_meas = r.shape[0]

print("Number of measurements = {}".format(num_meas))

'''*****************************************************************************
(2) Coordinate Transformation
*****************************************************************************'''
# Mapper  ->  Computations
#   x     ->     -y
#   y     ->      z
#   z     ->     -x
#Voltages:
# Mapper  ->  Computations
#   Ux     ->      -Ux'
#   Uy     ->       Uz'
#   Uz     ->       Uy'

#Orientation of sensors in sensor coordinates
N_s = np.eye(3)

#Transformation matrix mapper -> cone
R_cm = np.array([[ 0 , -1 , 0],
                 [ 0 ,  0 , 1],
                 [-1 ,  0 , 0]])

r = r @ R_cm

#Transformation matrix sensor -> cone
R_cs = np.array([[-1 ,  0 , 0],
                 [ 0 ,  0 , 1],
                 [ 0 ,  1 , 0]])

#Orientation of sensors in computation coordinates
N_c = N_s @ R_cs

nx_c = N_c[0,:]
ny_c = N_c[1,:]
nz_c = N_c[2,:]

#Voltages in computational coordinates
U_c = U @ R_cs

print('Sensor Orientations Cone Coordinates:')
print('n_x = {}'.format(nx_c))
print('n_y = {}'.format(ny_c))
print('n_z = {}'.format(nz_c))



fig = plt.figure()
ax = fig.gca(projection='3d')
sh.sh_quiver_plot(fig,ax,r[:,0]*1e3, r[:,1]*1e3, r[:,2]*1e3, U_c[:,0], U_c[:,1], U_c[:,2],cbarlabel=r'$U_\mathrm{Hall}\rightarrow$ [V]')
plt.show()

'''*****************************************************************************
(3) Preparation
*****************************************************************************'''
#Measurements in row vector
y = U.T.flatten()

#gradient estimated from simulations
g_x = 0.01217342*1e3   #[T/m]
#Y_2^0 coefficient estimated from gradient
c_1 = 2*g_x*np.sqrt(np.pi/5)


#Field coefficiets
# c = ( c_10 , c_20, c_30, c_40, ...)
#Notice that c_10 is not allowed in a cone quadrupole!
# c_10: linear varying potential -> constant field
c = np.array([0.,c_1,0,100000.,0.,0.,0.,0.])

#These are the designed sensor positions (in sensor coordinates)
offs_x = -2e-3            #+/- 0.2 [mm]
offs_y = -2e-3           #+/- 0.2 [mm]
offs_z =  0.2e-3           #+/- 0.2 [mm]

d_mat = np.array([[ offs_x ,    0.   ,    0.   ],
                  [  0.    ,  offs_y ,    0.   ],
                  [  0.   ,     0.    ,  offs_z ]])

#rotate to the measurement frame
d_mat =  d_mat @ R_cs


#these are the probe positions
d_x = d_mat[0,:]
d_y = d_mat[1,:]
d_z = d_mat[2,:]

#transfer_functions
s_x = np.array([5.01181543115734,-0.000667243])
s_y = np.array([4.96738878778532,0.000697497])
s_z = np.array([4.9593439758002 ,-0.0009172])


#and these are the sensor orientations
beta_x =  -1.*np.arcsin(nx_c[2])
gamma_x = np.arccos(nx_c[0]/np.cos(beta_x))

alpha_y =  np.arcsin(ny_c[2])
gamma_y = -1.*np.arcsin(ny_c[0]/np.cos(alpha_y))

alpha_z = -1.*np.arcsin(nz_c[1])
beta_z = np.arcsin(nz_c[0]/np.cos(alpha_z))


print('Initial Angles in Cone Coordinates:')
print('beta_x = {} deg.'.format(beta_x*180/np.pi))
print('gamma_x = {} deg.'.format(gamma_x*180/np.pi))

print('alpha_y = {} deg.'.format(alpha_y*180/np.pi))
print('gamma_y = {} deg.'.format(gamma_y*180/np.pi))

print('alpha_z = {} deg.'.format(alpha_z*180/np.pi))
print('beta_z = {} deg.'.format(beta_z*180/np.pi))


#measurement std
sigma_u = 1e-3  #[V]

#measurement precision
l_u = 1/sigma_u**2

'''*****************************************************************************
(4) Prior
*****************************************************************************'''
#maximum order of solid harmonics
L = len(c)-1

x_0 = np.zeros((L+15))
x_0[0:L] = c[1:]
x_0[L:L+6] = np.array([alpha_y,alpha_z,beta_x,beta_z,gamma_x,gamma_y])
x_0[L+6:L+15] = np.array([d_x[0],d_x[1],d_x[2],
                            d_y[0],d_y[1],d_y[2],
                            d_z[0],d_z[1],d_z[2]])

#number of unknowns
N = len(x_0)


'''*****************************************************************************
Model
*****************************************************************************'''
M = num_meas*3
Nrand = M+N


def A(x):
    #The dipole component is set to zero
    tmp = np.zeros((L+1,))
    tmp[1:] = x[:L]

    return np.sqrt(l_u)*sh.M_sym(r,tmp,x[L],x[L+1],x[L+2],x[L+3],x[L+4],x[L+5],x[L+6:L+9],x[L+9:L+12],x[L+12:L+15],s_x,s_y,s_z)

#right hand side
b = np.sqrt(l_u)*y

def Jac(x):

    J = np.zeros((M,N))

    tmp = np.zeros((L+1,))
    tmp[1:] = x[:L]

    tmp_Jac = sh.Jac_M_sym(r,tmp,x[L],x[L+1],x[L+2],x[L+3],x[L+4],x[L+5],x[L+6:L+9],x[L+9:L+12],x[L+12:L+15],s_x,s_y,s_z)
    J[:M,:] = np.sqrt(l_u)*np.delete(tmp_Jac,[0,L+16,L+17,L+18],1)#

    #minus because here the residual is defined as A-b!
    return -1*J


def qt_res_J(x,Q,e):

    r = A(x) - b

    Qtr = Q.T @ (r - e)

    J = Jac(x)
    QtJ = Q.T @ J

    return Qtr,QtJ,r

def residual_MAP(x):
    r = A(x) -b

    return r

def jacobian_MAP(x):

    return  Jac(x)

def residual(x,Q,e):
    r = A(x) - b
    Qtr = Q.T @ (r - e)

    return Qtr

def jacobian(x,Q,e):

    J = Jac(x)
    QtJ = Q.T @ J

    return QtJ

def print_result(res,L):

    print('\nSolution:')
    print('Field:')
    for l in range(L):
        print('\tc{} = {} T/m^{}'.format(l+1,res.x[l]*1e3**(l+1),l+1))

    print('Angles:')
    print('\tbeta_x = {} deg.'.format((res.x[L+2])*180/np.pi))
    print('\tgamma_x = {} deg.'.format((res.x[L+4])*180/np.pi))

    print('\talpha_y = {} deg.'.format((res.x[L])*180/np.pi))
    print('\tgamma_y = {} deg.'.format((res.x[L+5])*180/np.pi))

    print('\talpha_z = {} deg.'.format((res.x[L+1])*180/np.pi))
    print('\tbeta_z = {} deg.'.format((res.x[L+3])*180/np.pi))

    print('Position:')
    print('\tdx = {} mm'.format(res.x[L+6:L+9]))
    print('\tdy = {} mm'.format(res.x[L+9:L+12]))
    print('\tdz = {} mm'.format(res.x[L+12:L+15]))

def evaluate_solution(x):
    tmp = np.zeros((L+1,))
    tmp[1:] = x[:L]

    y_pred = sh.M_sym(r,tmp,                                        #field DoFs
                        x[L],                                #alpha_y
                          x[L+1],                            #alpha_z
                          x[L+2],                            #beta_y
                          x[L+3],                            #beta_z
                          x[L+4],                            #gamma_x
                          x[L+5],                            #gamma_y
                          x[L+6:L+9],                        #dx
                          x[L+9:L+12],                       #dy
                          x[L+12:L+15],                      #dz
                          s_x,s_y,s_z)

    return y_pred
'''*****************************************************************************
Test Levenberg-Marquardt Parameter Estimation
*****************************************************************************'''
tmp = np.zeros((L+1,))
tmp[1:] = x_0[:L]

y_pred_0 = sh.M_sym(r,tmp,x_0[L],x_0[L+1],x_0[L+2],x_0[L+3],x_0[L+4],x_0[L+5],x_0[L+6:L+9],x_0[L+9:L+12],x_0[L+12:L+15],s_x,s_y,s_z)

fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111)
ax.plot(y ,label = 'Measured')
ax.plot(y_pred_0,label = 'Perfect Cone Quad')
ax.legend()
ax.grid(which='both')
ax.set_xlabel(r'sample $\rightarrow$')
ax.set_ylabel(r'Voltage $\rightarrow$ [V]')
plt.show()

print('Compute MAP solution')
start_t = time.time()
res_map = least_squares(residual_MAP, x_0,jac = jacobian_MAP, verbose=2)# ,jac = jacobian_MAP  method = 'lm',
stop_t = time.time()


print('Elapsed time for MAP solution = {} sec.'.format(stop_t-start_t))

print_result(res_map,L)

y_pred = evaluate_solution(res_map.x)

xMAP = res_map.x
rMAP = res_map.fun

mse = np.sqrt(np.mean((y_pred-y)**2))

print('Measn squared error = {} V'.format(mse))


fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(211)
ax.plot(y,label = 'Measured')
ax.plot(y_pred_0,label = 'x_0')
ax.plot(y_pred,label = 'x_MAP')
ax.legend()
ax.grid(which='both')
ax.set_xlabel(r'sample $\rightarrow$')
ax.set_ylabel(r'Voltage $\rightarrow$ [V]')
ax = fig.add_subplot(212)
ax.plot(y-y_pred,label = 'difference')
ax.legend()
ax.grid(which='both')
ax.set_xlabel(r'sample $\rightarrow$')
ax.set_ylabel(r'Voltage $\rightarrow$ [V]')
plt.show()

'''*****************************************************************************
Linear Uncertainty Quantification
*****************************************************************************'''
dH = Jac(xMAP)/np.sqrt(l_u)

P_inv = l_u*dH.T @ dH
P = np.linalg.inv(P_inv)

P_lin_df = pd.DataFrame(data = P)
P_lin_df.to_csv('results/P_lin.csv',index=False)

'''*****************************************************************************
Transform to mapper coordinates
*****************************************************************************'''

#angles in cone coordinates
alpha_y = xMAP[L],                              #alpha_y
alpha_z = xMAP[L+1],                            #alpha_z
beta_x  = xMAP[L+2],                            #beta_y
beta_z  = xMAP[L+3],                            #beta_z
gamma_x = xMAP[L+4],                            #gamma_x
gamma_y = xMAP[L+5],                            #gamma_y

#positions in cone coordinates
dx = xMAP[L+6:L+9]     #dx
dy = xMAP[L+9:L+12]    #dy
dz = xMAP[L+12:L+15]   #dz

#these are the orientation vectors in cone magnet coordinates
nx = np.array([np.cos(gamma_x)*np.cos(beta_x),
                   np.sin(gamma_x)*np.cos(beta_x),
                  -np.sin(beta_x)])

ny = np.array([-np.sin(gamma_y)*np.cos(alpha_y),
                   np.cos(gamma_y)*np.cos(alpha_y),
                   np.sin(alpha_y)])

nz = np.array([np.sin(beta_z)*np.cos(alpha_z),
                   -np.sin(alpha_z),
                   np.cos(beta_z)*np.cos(alpha_z)])

#these are the relative angles
phi_xy = np.arccos(nx.T @ ny)[0,0]
phi_xz = np.arccos(nx.T @ nz)[0,0]
phi_yz = np.arccos(ny.T @ nz)[0,0]

print('phi_xy = {} deg.'.format(phi_xy*180/np.pi))
print('phi_xz = {} deg.'.format(phi_xz*180/np.pi))
print('phi_yz = {} deg.'.format(phi_yz*180/np.pi))

#these are the orientation vectors in mapper coordinates
nx_m = nx.T @ R_cm.T
ny_m = ny.T @ R_cm.T
nz_m = nz.T @ R_cm.T

print('Results in mapper coordinates')

print('Orientation vectors')
print('nx = {}'.format(nx_m))
print('ny = {}'.format(ny_m))
print('nz = {}'.format(nz_m))

#these are the positions in mapper coordinates
dx_m = dx.T @ R_cm.T
dy_m = dy.T @ R_cm.T
dz_m = dz.T @ R_cm.T

print('Probe positions')
print('pos y = {} mm'.format(dy_m*1e3))
print('dx - dy = {} mm'.format((dx_m - dy_m)*1e3))
print('dz - dy = {} mm'.format((dz_m - dy_m)*1e3))



if enable_MCMC:

    '''*****************************************************************************
    Metropolis Hastings RTO proposal
    *****************************************************************************'''

    def RTO_MH(x0,cost,Q,nsamp):

        Nrand = M
        xchain = np.zeros((N,nsamp+1))
        xchain[:,0] = x0
        e = np.zeros((Nrand,))   #eps = 0

        Qtr,QtJ,r = qt_res_J(x0,Q,e)

        log_c_chain = np.zeros((nsamp+1,1))
        log_c_chain[0] = np.sum(np.log(np.diag(sci.linalg.cholesky(QtJ.T @ QtJ)))) \
                        + 0.5*(np.linalg.norm(r)**2 - np.linalg.norm(Qtr)**2)

        naccept = 0

        for i in range(nsamp):

            e = np.random.randn(Nrand)   #eps ~ N(0,1)

            params = (Q,e)
            res_ls = least_squares(residual, x0, jac = jacobian, args=params,xtol=1e-14,gtol=1e-12, method = 'lm',ftol=1e-15)#

            #print(p[4])
            #print(res_ls.fun)
            #print(residual(res_ls.x,p[0],p[1],p[2],p[3],p[4],p[5],e,p[7]))

            #exit()
            nresid = np.linalg.norm(res_ls.fun)**2

            e = np.zeros((Nrand,))   #eps = 0

            Qtr,QtJ,r = qt_res_J(x0,Q,e)
            log_c_tmp = np.sum(np.log(np.diag(sci.linalg.cholesky(QtJ.T @ QtJ)))) \
                            + 0.5*(np.linalg.norm(r)**2 - np.linalg.norm(Qtr)**2)

            #print(nresid)
            if log_c_chain[0] - log_c_tmp > np.log(np.random.rand(1)[0]) and nresid < 1e-8:
                #print('accept')
                naccept = naccept + 1
                xchain[:,i+1] = res_ls.x
                log_c_chain[i+1] = log_c_tmp

            else:
                #print('reject')
                xchain[:,i+1] = xchain[:,i]
                log_c_chain[i+1] = log_c_chain[i]


        return xchain[:,1:],naccept/nsamp,log_c_chain


    nsamps = 10
    rsamp = rMAP

    x_i = xMAP.copy()


    ffile = open('results/samples.txt',"w+")
    for l in range(L):
        ffile.write("c_{},".format(l+1))
    ffile.write("alpha_y,alpha_z,beta_x,beta_z,gamma_x,gamma_y,")
    ffile.write("dx_x,dx_y,dx_z,")
    ffile.write("dy_x,dy_y,dy_z,")
    ffile.write("dz_x,dz_y,dz_z")
    ffile.write("\n")
    ffile.close()


    start_t = time.time()
    for i in range(nsamps-1):

        print('Iteration {}'.format(i))
        #sample lambda
        resid = y-A(x_i)

        res_map = least_squares(residual_MAP, xMAP,jac = jacobian_MAP)#, method = 'lm'

        #print('\txMAP ready')
        #print('Angles MAP = {} deg.'.format(res_map.x[-6:]*180/np.pi))
        #print('Angular Errors = {} urad'.format((theta_gt-res_map.x[-6:])*1e6))

        Q,_ = sci.linalg.qr(res_map.jac,mode='economic')
        #print(res_map.jac.shape)
        #print(Q.shape)

        #params = (A,b,lamsamp[i+1],delsamp[i+1],Q,np.zeros((Nrand,)),Nrand)

        x_tmp,_,_ = RTO_MH(xMAP,residual,Q,1)


        x_i = x_tmp[:,-1].copy()
        #xchain[:,i+1] =
        #Axchain[:,i+1] = A(x_tmp[:,-1])

        #print('Angles RTO = {} deg.'.format(xchain[-6:,i+1]*180/np.pi))

        for l in range(L):
            print('\tc{} = {} T/m^{}'.format(l+1,x_i[l]*1e3**(l+1),l+1))
        print('\tAngles = {} deg.'.format((x_i[L:L+6])*180/np.pi))
        print('\tdx = {} mm'.format(x_i[L+6:L+9]))
        print('\tdy = {} mm'.format(x_i[L+9:L+12]))
        print('\tdz = {} mm'.format(x_i[L+12:L+15]))


        ffile = open('results/samples.txt',"a")
        for j,xx in enumerate(x_i):
            if (j == 0):
                ffile.write("{}".format(xx))
            else:
                ffile.write(",{}".format(xx))
        ffile.write("\n")

        ffile.close()


    stop_t = time.time()
    print('Elapsed Time for {} samples = {} sec.'.format(nsamps,stop_t-start_t))
