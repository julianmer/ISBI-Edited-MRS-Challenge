####################################################################################################
#                                             basis.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 03/02/23                                                                                #
#                                                                                                  #
# Purpose: Defines a main structure for edited MRS metabolite basis sets. Encapsulates all         #
#          information and holds definitions to compute various aspects.                           #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np
import os
import torch

from scipy.io import loadmat



#**************************************************************************************************#
#                                          Class EdBasis                                           #
#**************************************************************************************************#
#                                                                                                  #
# The main structure for edited MRS metabolite basis sets.                                         #
#                                                                                                  #
#**************************************************************************************************#
class EdBasis():

    #*************************#
    #   initialize instance   #
    #*************************#
    def __init__(self, path2basis):
        """
        Main init for the MyMRS class.

        @param path2basis -- The path to the basis set folder.
        """
        names = sorted(os.listdir(path2basis))

        self.names = []
        fidsON = []
        specsON = []
        fidsOFF = []
        specsOFF = []
        for name in names:
            if 'ON' in name:
                self.names.append(name[:-7])
                data = loadmat(path2basis + '/' + name)
                fidsON.append(data['fids'])
                specsON.append(data['specs'])
            elif 'OFF' in name:
                data = loadmat(path2basis + '/' + name)
                fidsOFF.append(data['fids'])
                specsOFF.append(data['specs'])

        self.fidsON = np.swapaxes(np.array(fidsON).squeeze(), 0, 1)
        self.fidsOFF = np.swapaxes(np.array(fidsOFF).squeeze(), 0, 1)
        self.specsON = np.swapaxes(np.array(specsON).squeeze(), 0, 1)
        self.specsOFF = np.swapaxes(np.array(specsOFF).squeeze(), 0, 1)

        # conjugate
        self.fidsON = np.conjugate(self.fidsON)
        self.fidsOFF = np.conjugate(self.fidsOFF)
        self.specsON = np.conjugate(self.specsON)
        self.specsOFF = np.conjugate(self.specsOFF)

        try:
            self.bw = int(data['spectralwidth'])
            self.dwelltime = data['dwelltime']
            self.n = int(data['n'])
            self.t = data['t']
            self.f = torch.arange(- self.bw / 2, self.bw / 2, self.bw / self.n)
            self.ppm = data['ppm'].squeeze()
            self.cf = data['txfrq']
        except:
            print('Given directory is empty!')