
import h5py as h5
import numpy as np
import json, os

def save(L, A, M, Tr, dt, XU, XUt, G, folder='data/', 
         CHECK_IF_SAVE_EXISTS=True, PRINT_SAVE=True):
    """
    Create a folder with a H5 file for data and JSON for the parameters.
    """
    # get folder name
    path = get_datafolder_path(L, A, M, Tr, dt, folder)

    data_path = path + '/data.h5'
    param_path = path + '/params.json'
    
    if os.path.isdir(path): # check that folder doesn't exist
        if CHECK_IF_SAVE_EXISTS:
            ans = input('Folder already exists. Overwrite previous data [y/n]? ')
            if ans == 'n':
                print('Not saving data.')
                return

        # delete old files
        os.remove(data_path)
        os.remove(param_path)
    else:
        os.mkdir(path)
    
    f = h5.File(data_path, 'w')
    f.create_dataset('G', data=G)
    f.create_dataset('XU', data=XU)
    f.create_dataset('XUt', data=XUt)
    f.close()
    
    param_dict = {'SIGMA': float(L[0]), 
                  'RHO'  : float(L[1]), 
                  'BETA' : float(L[2]),
                  'sigma': float(A[0]), 
                  'beta' : float(A[1]), 
                  'rho'  : float(A[2]),
                  'mu1'  : int(M[0]), 
                  'mu2'  : int(M[1]), 
                  'mu3'  : int(M[2]),
                  'Tr'   : float(Tr), 
                  'dt'   : float(dt)}
    with open(param_path, 'w') as out:
        json.dump(param_dict, out)
     
    if PRINT_SAVE:
        print('Saved data and parameters to:', path + '/')

def get_datafolder_path(L, A, M, Tr, dt, folder):
    """
    Create a folder name based on the system and algorithm parameters.
    """
    fname =  'S_{:.0f}_R_{:.0f}_B_{:.2f}_'.format(*L)
    fname += 's_{:.4f}_r_{:.4f}_b_{:.4f}_'.format(*A)
    fname += 'mu1_{:.0f}_mu2_{:.0f}_mu3_{:.0f}_'.format(*M)
    fname += 'Tr_{:}_dt_{:}'.format(Tr, dt)
    
    # append parent folder
    if len(folder) > 0 and folder[-1] != '/': 
        folder += '/'

    return folder + fname

def check_sim_exists(L, A, M, Tr, dt, folder):
    path = get_datafolder_path(L, A, M, Tr, dt, folder)
    if os.path.isdir(path):
        return path
    else:
        return None