import numpy as np
import spherical_harmonics as sh
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import gamma as gamma_distr
from scipy.optimize import least_squares

class MCMC_RTO:
    def __init__(self,y,R,x_0,L_0,ad_0,bd_0,F,JacF):

        self.set_prior_delta(ad_0,bd_0)
        self.set_prior_x(x_0,L_0)
        self.set_forward_operator(F,JacF)
        self.set_measurements(y,R)



    def set_measurements(self,pos,y,R,l=1.):
        self.pos = pos
        self.y = y
        self.R = R
        self.num_meas = len(y)
        self.llambda = l

    def set_prior_delta(self,ad_0,bd_0):
        self.ad_0 = ad_0
        self.bd_0 = bd_0

    def set_prior_x(self,x_0,L_0):
        self.x_0 = x_0
        self.L_0 = L_0
        self.L_sqrt = np.sqrt(L_0)  #Cholesky if L_0 not diagonal
        self.num_dofs = len(x_0)


    def set_forward_operator(self,F,JacF):
        self.F = F
        self.J = JacF

    def A_d(self,x,d):
        A = np.zeros((self.num_meas+self.num_dofs,))
        A[:self.num_meas] = self.F(x)
        A[self.num_meas:] = np.sqrt(d)*self.L_sqrt @ x

        return A

    def b_d(self,d):
        b = np.zeros((self.num_meas+self.num_dofs,))
        b[:self.num_meas] = np.sqrt(self.llambda)*self.y
        b[self.num_meas:] = np.sqrt(d)*self.L_sqrt @ x_0

        return b

    def r_d(self,x,d):
        return self.A_d(x,d) - self.b_d(d)

    def obj_fcn_map(self,x,d):
        return self.A_d(x,d) - self.b_d(d)

    def obj_fcn_rto(self,x,d,Q,eps):
            return np.dot(Q.T,(self.A_d(x,d)-(self.b_d(d) + eps)))

    def Jac_d(self,x,d):
        J = np.zeros((self.num_meas+self.num_dofs,self.num_dofs))

        J[:self.num_meas,:] = np.diag(np.sqrt(self.llambda)) @ self.J(self.pos,x)

        J[self.num_meas:,:] = np.sqrt(d)*self.L_sqrt

        return J

    def Jac_rto(self,x,d,Q,eps):

        return Q.T @ self.Jac_d(x,d)


if __name__ == '__main__':

    #We test the MCMC RTO

    #Number of measurements along one dimension
    num_meas = 10
    #We make up measurements on a 3d grid
    X,Y,Z = np.meshgrid(np.linspace(-3,3.,num_meas),
                       np.linspace(-3,3.,num_meas),
                       np.linspace(-3,3.,num_meas))

    #Measurement positions
    r = np.zeros((X.flatten().shape[0],3))
    r[:,0] = X.flatten()
    r[:,1] = Y.flatten()
    r[:,2] = Z.flatten()

    #These are the sensor positions
    dx = np.array([-2.125 ,    1.  , 0.4  ])
    dy = np.array([   0.1  , -2.125 , 0.  ])
    dz = np.array([   2.  ,    -0.2  , -0.1])

    #and these are the sensor orientations
    alpha_y =  1*np.pi/180
    alpha_z = -1*np.pi/180

    beta_x =    2*np.pi/180
    beta_z = -0.5*np.pi/180

    gamma_x =  0.5*np.pi/180
    gamma_y = -1.5*np.pi/180

    #sensitivity
    sx = 5.1
    sy = 4.9
    sz = 5.2

    #Field coefficiets
    c = np.array([0,10,0.02,0.001,0.0003])

    #compute measurements
    y = sh.M(r,c,alpha_y,alpha_z,beta_x,beta_z,gamma_x,gamma_y,dx,dy,dz,sx,sy,sz)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(y,label='y')
    plt.show()


    #Add some noise
    n_std = 1e-3
    n = n_std*np.random.randn(y.shape[0])
    y += n

    #Measurement noise covariance
    R = n_std**2*np.eye(len(y))

    #Plot voltages
    Ux = y[:num_meas**3]
    Uy = y[num_meas**3:2*num_meas**3]
    Uz = y[2*num_meas**3:]

    Ux.shape = X.shape
    Uy.shape = X.shape
    Uz.shape = X.shape

    C = np.sqrt(Ux**2+Uy**2+Uz**2)

    minima = min(C.flatten())
    maxima = max(C.flatten())

    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.jet)

    fig = plt.figure(figsize=(15,10))
    ax = fig.gca(projection='3d')
    quiv = ax.quiver(X, Y, Z, Ux, Uy, Uz,color=mapper.to_rgba(C.flatten()) ,length=0.005,cmap = 'jet')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    fig.colorbar(quiv)
    plt.show()

    #prior
    #             [-  -c  -  ] [  angles   ]  [  dx   ] [  dy   ] [  dz   ]  [sz]
    x_0 = np.array([10,0,0,0 , 0,0,0,0,0,0,  -2,0,0,   0,-2,0,   0,0,-0.2   , 5])

    #Prior uncertainty
    std_c = 1
    std_angles = 3*180/np.pi
    std_pos = 2
    std_sz = 0.1

    L_0 = np.diag(1/np.array([1/std_c**2,1/std_c**2,1/std_c**2,1/std_c**2,
                1/std_angles**2,1/std_angles**2,1/std_angles**2,1/std_angles**2,1/std_angles**2,1/std_angles**2,
                1/std_pos**2,1/std_pos**2,1/std_pos**2,1/std_pos**2,1/std_pos**2,1/std_pos**2,1/std_pos**2,1/std_pos**2,1/std_pos**2,
                1/std_sz**2]))

    #*******************************************
    #Least Squares Solution
    #*******************************************
    #objective function to minimize
    def residuals(x):
        #The dipole component is set to zero
        tmp = np.zeros((5,))
        tmp[1:] = x[:4]
        return y - sh.M(r,tmp,x[4],x[5],x[6],x[7],x[8],x[9],x[10:13],x[13:16],x[16:19],sx,sy,x[19])

    #in the jacobian, we need to filter out the columns corresponding to known parameters
    def jacobian(x):
        tmp = np.zeros((5,))
        tmp[1:] = x[:4]
        J = sh.Jac_M(r,tmp,x[4],x[5],x[6],x[7],x[8],x[9],x[10:13],x[13:16],x[16:19],sx,sy,x[19])

        return np.delete(J,[0,20,21],1)

    ls_res = least_squares(residuals, x_0,jac=jacobian, verbose=2)

    print('Errors least squares solution')

    print('Field:')
    print('\tc1 = {}'.format(ls_res.x[0]-c[1]))
    print('\tc2 = {}'.format(ls_res.x[1]-c[2]))
    print('\tc3 = {}'.format(ls_res.x[2]-c[3]))
    print('\tc4 = {}'.format(ls_res.x[3]-c[4]))

    print('Angles:')
    print('\talpha_y = {} deg.'.format((ls_res.x[4]-alpha_y)*180/np.pi))
    print('\talpha_z = {} deg.'.format((ls_res.x[5]-alpha_z)*180/np.pi))
    print('\tbeta_x = {} deg.'.format((ls_res.x[6]-beta_x)*180/np.pi))
    print('\tbeta_z = {} deg.'.format((ls_res.x[7]-beta_z)*180/np.pi))
    print('\tgamma_x = {} deg.'.format((ls_res.x[8]-gamma_x)*180/np.pi))
    print('\tgamma_y = {} deg.'.format((ls_res.x[9]-gamma_y)*180/np.pi))

    print('Position:')
    print('\tdx = {}'.format(ls_res.x[10:13]-dx))
    print('\tdy = {}'.format(ls_res.x[13:16]-dy))
    print('\tdz = {}'.format(ls_res.x[16:19]-dz))

    print('Sensitivity:')
    print('\tsz = {}'.format(ls_res.x[19]-sz))

    '''
    #prior regularization parameter
    d_0 = 1e-3
    #We want a flat prior for delta
    delta_std = 0.9*d_0
    delta_var = delta_std**2

    bd_0 = d_0/delta_var
    ad_0 = bd_0**2*delta_var

    s = gamma_distr.rvs(ad_0,scale=1/bd_0,size=1000) #np.random.gamma(alpha_0,beta_0,1000)

    print('Mean delta = {}'.format(np.mean(s)))
    print('Var delta = {}'.format(np.std(s)**2))
    print('STD delta = {}'.format(np.std(s)))

    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(1,1,1)
    ax.hist(s,bins=150)
    ax.set_ylabel('frequency')
    ax.set_xlabel(r'$\delta \rightarrow$')
    plt.show()

    #setup MCMC RTO
    sampler = MCMC_RTO(y,R,x_0,L_0,ad_0,bd_0,F,JacF)
    '''
