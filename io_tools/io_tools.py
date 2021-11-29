import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""*****************************************************************************
    Read measurement data
*****************************************************************************"""
def read_measurement_data(meas_filename,v = 0.25e-3,
                                        aper = 0.0005,
                                        dir = np.array([0.,0.,1.]),
                                        center = np.array([0.,0.,0.]),
                                        filter = 3):

    meas_data = pd.read_csv(meas_filename,sep='\t')
    #filter out no samples
    def isnumber(x):
        try:
            float(x)

            return True
        except:
            return False

    meas_data = meas_data[meas_data.applymap(isnumber)].astype(float)

    (num_steps,num_moves) = meas_data.values.shape
    num_moves = np.int32(num_moves/7)

    r = np.zeros((num_steps*num_moves,3))
    U = np.zeros((num_steps*num_moves,3))

    k = 0

    #We perform all the computations in mm
    for i in range(num_moves):

        r[k*num_steps:(k+1)*num_steps,0] = meas_data.values[:,i*7]*1e-3
        r[k*num_steps:(k+1)*num_steps,1] = meas_data.values[:,i*7+1]*1e-3
        r[k*num_steps:(k+1)*num_steps,2] = meas_data.values[:,i*7+2]*1e-3

        if(i % 2 != 0):
            r[k*num_steps:(k+1)*num_steps,0] -= dir[0]*aper*v/2.
            r[k*num_steps:(k+1)*num_steps,1] -= dir[1]*aper*v/2.
            r[k*num_steps:(k+1)*num_steps,2] -= dir[2]*aper*v/2.
        else:
            r[k*num_steps:(k+1)*num_steps,0] += dir[0]*aper*v/2.
            r[k*num_steps:(k+1)*num_steps,1] += dir[1]*aper*v/2.
            r[k*num_steps:(k+1)*num_steps,2] += dir[2]*aper*v/2.

        U[k*num_steps:(k+1)*num_steps,0] = meas_data.values[:,i*7+3]
        U[k*num_steps:(k+1)*num_steps,1] = meas_data.values[:,i*7+4]
        U[k*num_steps:(k+1)*num_steps,2] = meas_data.values[:,i*7+5]

        if(False):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(r[k*num_steps:(k+1)*num_steps,2],U[k*num_steps:(k+1)*num_steps,0],label=r'$U_x$')
            ax.plot(r[k*num_steps:(k+1)*num_steps,2],U[k*num_steps:(k+1)*num_steps,1],label=r'$U_y$')
            ax.plot(r[k*num_steps:(k+1)*num_steps,2],U[k*num_steps:(k+1)*num_steps,2],label=r'$U_z$')
            ax.legend()
            plt.show()

        k += 1

    #delete nans
    del_list = []

    for k,uu in enumerate(U):
        if any(np.isnan(uu)):
            #print('delete {}'.format(k))
            del_list.append(k)

    r = np.delete(r,del_list,axis=0)
    U = np.delete(U,del_list,axis=0)


    #estimated field zero
    r_0 = np.mean(r,axis=0)
    U_0 = np.mean(U,axis=0)



    print('Coordinate center = {}'.format(r_0))

    r[:,0] -= r_0[0]
    r[:,1] -= r_0[1]
    r[:,2] -= r_0[2]


    x_max = max(r[:,0])
    x_min = min(r[:,0])

    y_max = max(r[:,1])
    y_min = min(r[:,1])

    z_max = max(r[:,2])
    z_min = min(r[:,2])

    print('x_max = {} mm'.format(x_max*1000))
    print('x_min = {} mm'.format(x_min*1000))
    print('y_max = {} mm'.format(y_max*1000))
    print('y_min = {} mm'.format(y_min*1000))
    print('z_max = {} mm'.format(z_max*1000))
    print('z_min = {} mm'.format(z_min*1000))

    #sphere radius
    R = 0.5*(y_max-y_min)

    #filter out sphere
    U = U[(r[:,0]-center[0])**2+(r[:,1]-center[1])**2+(r[:,2]-center[2])**2 <= R**2,:]
    r = r[(r[:,0]-center[0])**2+(r[:,1]-center[1])**2+(r[:,2]-center[2])**2 <= R**2,:]

    r = r[::filter,:]
    U = U[::filter,:]

    num_meas = r.shape[0]

    print('Number of measured points = {}'.format(num_meas))

    return r,U,r_0
