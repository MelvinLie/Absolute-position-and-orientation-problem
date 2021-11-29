import numpy as np
from scipy.special import sph_harm
import matplotlib.pyplot as plt
from scipy.special import lpmn
from scipy.special import factorial
import numpy.matlib
import matplotlib.cm as cm
import matplotlib

'''
Function to plot vector fields nicely in 3D
'''
def sh_quiver_plot(fig,ax,x,y,z,ux,uy,uz,cbarlabel='',length=1.):

    c = np.sqrt(ux**2+uy**2+uz**2)
    tmp = np.matlib.repmat(c,2,1)
    c = np.append(c,tmp.T.flatten())



    minima = min(c)
    maxima = max(c)

    norm = matplotlib.colors.Normalize(vmin=minima, vmax=maxima, clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=cm.jet)

    #ax = fig.gca(projection='3d')
    quiv = ax.quiver(x,y,z,ux,uy,uz,colors = mapper.to_rgba(c),length=length)
    cbar = plt.colorbar(mapper)
    cbar.set_label(cbarlabel,fontsize=20)
    cbar.ax.tick_params(labelsize=16)
    return quiv

'''
This function computes the derivative of a spherical harmonic with respect to $\theta$
'''
def dYdt(m,l,p,t):

    #Derivatives of associated Legendre polynomials are infinity in pole $\theta = 0$
    threshold = 1e-5
    t[t < threshold] = threshold
    t[abs(t-np.pi) < threshold] = np.pi-threshold

    ret_val = m/np.tan(t)*sph_harm(m,l,p,t)

    cos_t = np.cos(t)

    ret_val = -np.array([lpmn(m,l,c_t)[-1][-1,-1] for c_t in cos_t])*np.sin(t)+0j

    ret_val *= np.exp(1j*m*p)
    ret_val *= np.sqrt((2*l+1)*factorial(l-m)/4/np.pi/factorial(l+m))


    return ret_val

'''
This function computes the second derivative of a spherical harmonic with respect to $\theta$
'''
def d2Ydt2(m,l,p,t):

    #Derivatives of associated Legendre polynomials are infinity in pole $\theta = 0$
    threshold = 1e-5
    t[t < threshold] = threshold
    t[abs(t-np.pi) < threshold] = np.pi-threshold

    Yml = sph_harm(m,l,p,t)
    if(m+1 > l) : Ymp1_l = 0.
    else: Ymp1_l = sph_harm(m+1,l,p,t)
    if(m+2 > l): Ymp2_l = 0
    else: Ymp2_l = sph_harm(m+2,l,p,t)

    cot_t = 1/np.tan(t)
    if ( (l-m-1)*(l+m+2) > 0):
        sqrt_tmp = np.sqrt((l-m-1)*(l+m+2))
    else:
        sqrt_tmp = np.sqrt(abs((l-m-1)*(l+m+2)))*1j


    #return symbolic_second_derivative.evalf(subs={m_s: m, l_s: l, phi_s: p, theta_s: t})
    A = m*cot_t*Yml
    B = np.sqrt((l-m)*(l+m+1))*np.exp(-1j*p)*Ymp1_l
    C = m*(-cot_t**2-1)*Yml
    D = sqrt_tmp*np.exp(-1j*p)*Ymp2_l#sph_harm(m+2,l,p,t) #np.sqrt((l-m-1)*(l+m+2))
    E = (m+1)*cot_t*Ymp1_l


    return m*(A + B)*cot_t + C + np.sqrt((l-m)*(l+m+1))*(D + E)*np.exp(-1j*p)
