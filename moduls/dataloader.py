####################################################################################################
#                                         dataloader.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl),                                                   #
#          S. Amirrajab (s.amirrajab@tue.nl)                                                       #
#                                                                                                  #
# Created: 23/01/23                                                                                #
#                                                                                                  #
# Purpose: Definition of data sets, taking care of loading and processing the data.                #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import h5py
import math
import numpy as np
import scipy
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

# own
from challenge import spec_reg
from simulation.basis import EdBasis
from simulation.sigModels import VoigtModel
from simulation.simulation import simulateParam
from simulation.simulationDefs import *


#**************************************************************************************************#
#                                        Class MasterDataset                                       #
#**************************************************************************************************#
#                                                                                                  #
# The main dataset class to inherit from for general loading and processing.                       #
#                                                                                                  #
#**************************************************************************************************#
class MasterDataset(Dataset):
    def __init__(self, data_dir, data_format='real', phase='train', register=False, transients=40):
        super(MasterDataset, self).__init__()

        self.data_dir = data_dir
        self.data_format = data_format   # real, imag, complex
        self.phase = phase
        self.register = register
        self.transients = transients   # number of transients

        self.setup(self.phase)   # to load  h5 data

    def setup(self, stage=None):
        pass

    def get_data_type(self, x, y):
        if self.data_format == 'real':
            x = np.real(x)
            y = np.real(y)
        elif self.data_format == 'complexX':
            x = x
            y = np.real(y)
        elif self.data_format == 'complex':
            x = x
            y = y
        else:
            ValueError ('real, imag, or complex for data_format' )
        return x, y

    def processData(self, xFID, yFID):
        """
            Transforms the input FIDs to the (subtracted) edited spectra and normalizes.

            :param xFID: The FID holding all transients, expected shape (points, 2, transients).
            :param yFID: The "GT" FID, expected shape (points, 2).
        """
        # transforming into frequency domain to use as the target data
        y = np.fft.fftshift(np.fft.ifft(yFID, axis=0), axes=0)
        y = y[..., 1] - y[..., 0]  # editing process in subtraction

        # transforming time domain noisy transients into frequency domain difference transients
        x = np.fft.fftshift(np.fft.ifft(xFID, axis=0), axes=0)

        x = x[..., 1, :] - x[..., 0, :]

        x, y = self.get_data_type(x, y)

        # normalize
        x /= np.max(np.abs(x), axis=0, keepdims=True)
        y /= np.max(np.abs(y), axis=0, keepdims=True)

        return x, y


#**************************************************************************************************#
#                                      Class SimulatedDataset                                      #
#**************************************************************************************************#
#                                                                                                  #
# Dataset used to create and process synthetic data.                                               #
#                                                                                                  #
#**************************************************************************************************#
class SimulatedDataset(MasterDataset):
    def __init__(self, transform=None, data_format='real', register=False, transients=40):
        self.transform = transform

        self.basisObj = EdBasis('./basis')
        self.params, self.concs = customParams, customConcs

        self.sigModelON = VoigtModel(self.basisObj.fidsON,
                                     first=0, last=self.basisObj.n,
                                     t=self.basisObj.t, f=self.basisObj.f)
        self.sigModelOFF = VoigtModel(basis=self.basisObj.fidsOFF,
                                      first=0, last=self.basisObj.n,
                                      t=self.basisObj.t, f=self.basisObj.f)

        MasterDataset.__init__(self, '', data_format, '', register, transients)

    def __len__(self):
        # return int(0.9 * 0.8 * 5000)
        return 1000

    def __getitem__(self, idx):
        n = len(self.basisObj.names)
        batch = 1

        thetaON, _, _ = simulateParam(self.basisObj, batch, self.params, self.concs)
        # thetaOFF, _, _ = simulateParam(self.basisObj, batch, self.params, self.concs)
        # thetaOFF = np.concatenate((thetaON[:, :n], thetaOFF[:, n:]), axis=1)
        thetaOFF = thetaON.copy()

        # distort signal params
        g = 1
        thetaOFF[:, :n] += 1e-1 * np.random.normal(0, 1, n)   # concentrations
        thetaOFF[:, n:n + g] += 1e-1 * np.random.normal(0, 1, g)    # lorentzian blurring
        thetaOFF[:, n + g:n + 2 * g] += 1e-1 * np.random.normal(0, 1, g)   # gaussian broadening
        thetaOFF[:, n + 2 * g:n + 3 * g] += np.random.normal(0, 1, g)   # frequency shift
        thetaOFF[:, n + 3 * g] += 1e-3 * np.random.normal(0, 1, 1)   # global phase shift
        thetaOFF[:, n + 3 * g + 1] += 1e-4 * np.random.normal(0, 0.1, 1)   # global phase ramp
        thetaOFF[:, n + 3 * g + 2:] += 1e-3 * np.random.normal(0, 1, thetaOFF[:, n + 3 * g + 2:].shape)   # baseline params

        specON = self.sigModelON.forward(torch.from_numpy(thetaON))
        specOFF = self.sigModelOFF.forward(torch.from_numpy(thetaOFF))

        spec = torch.cat((specOFF[..., None], specON[..., None]), dim=2)
        fid = np.fft.fft(spec, axis=1)

        x = fid.squeeze().copy()
        if self.transform is not None:
            for trans in self.transform:
                x = trans(x, self.basisObj.t.squeeze())
        else:
            x = x[..., np.newaxis]

        # correct phase and frequency
        if self.register:
            x = spec_reg.basic_spectral_registration(x[np.newaxis, ...],
                                                     self.basisObj.t).squeeze()
        # fft, subtraction, and normalization
        x, y = self.processData(x, fid.squeeze().copy())
        return x, y, self.basisObj.ppm.squeeze(), self.basisObj.t.squeeze()


#**************************************************************************************************#
#                                        Class SynthDataset                                        #
#**************************************************************************************************#
#                                                                                                  #
# Simple dataset used to load data, process and iterate over synthetic data.                       #
#                                                                                                  #
#**************************************************************************************************#
class SynthDataset(MasterDataset):
    def __init__(self,
                 data_dir='../../../data/Edited-MRS Reconstruction '
                          'Challenge Data/simulated_ground_truths.h5',
                 transform=None, data_format='real',
                 phase='train', register=False, transients=40):
        self.transform = transform
        MasterDataset.__init__(self, data_dir, data_format, phase, register, transients)

    def setup(self, stage=None):
        with h5py.File(self.data_dir) as hf:
            y = hf["ground_truth_fids"][()]
            ppm = hf["ppm"][()]
            t = hf["t"][()]
        self.y, self.ppm, self.t = y, ppm, t

        # division of training and testing data
        y_train = y[:int(y.shape[0] * 0.9)]
        y_test = y[int(y.shape[0] * 0.9):]

        ppm_train = ppm[:int(ppm.shape[0] * 0.9)]
        ppm_test = ppm[int(ppm.shape[0] * 0.9):]

        t_train = t[:int(t.shape[0] * 0.9)]
        t_test = t[int(t.shape[0] * 0.9):]

        # division of training and validation data
        y_val = y_train[int(y_train.shape[0] * 0.8):]
        y_train = y_train[:int(y_train.shape[0] * 0.8)]

        ppm_val = ppm_train[int(ppm_train.shape[0] * 0.8):]
        ppm_train = ppm_train[:int(ppm_train.shape[0] * 0.8)]

        t_val = t_train[int(t_train.shape[0] * 0.8):]
        t_train = t_train[:int(t_train.shape[0] * 0.8)]

        if stage == 'train': self.y, self.ppm, self.t = y_train, ppm_train, t_train
        elif stage == 'val': self.y, self.ppm, self.t = y_val, ppm_val, t_val
        elif stage == 'test': self.y, self.ppm, self.t = y_test, ppm_test, t_test

    def __len__(self):
        return int(self.y.shape[0])

    def __getitem__(self, idx):
        x = self.y[idx].copy()
        if self.transform is not None:
            for trans in self.transform:
                x = trans(x, self.t[idx])
        else:
            x = x[..., np.newaxis]

        # correct phase and frequency
        if self.register:
            x = spec_reg.basic_spectral_registration(x[np.newaxis, ...],
                                                     self.t[idx][np.newaxis, ...]).squeeze()
        # fft, subtraction, and normalization
        x, y = self.processData(x, self.y[idx].copy())
        return x, y, self.ppm[idx], self.t[idx]


#**************************************************************************************************#
#                                        Class Track2Dataset                                       #
#**************************************************************************************************#
#                                                                                                  #
# Dataset used to load data, process and iterate over the in-vivo data of track 2.                 #
#                                                                                                  #
#**************************************************************************************************#
class RealDataset(MasterDataset):
    def __init__(self,
                 data_dir='../../../data/Edited-MRS Reconstruction '
                          'Challenge Data/track_02_training_data.h5',
                 data_dir2='../../../data/Edited-MRS Reconstruction '
                           'Challenge Data/track_03_training_data.h5',
                 transform=None, data_format='real',  phase='train',
                 register=False, transients=40, track='track2', clean=False):
        self.transform = transform
        self.data_dir2 = data_dir2
        self.track = track
        self.clean = clean
        MasterDataset.__init__(self, data_dir, data_format, phase, register, transients)

    def setup(self, stage=None):
        if self.track == 'track2' or self.track == 'both':
            with h5py.File(self.data_dir) as hf:
                x = hf['transient_fids'][()]
                y = hf['target_spectra'][()]
                ppm = hf["ppm"][()]
                t = hf["t"][()]

        if self.track == 'track3' or self.track == 'both':
            with h5py.File(self.data_dir2) as hf2:
                data_2048 = hf2['data_2048']
                data_4096 = hf2['data_4096']

                x_2048 = data_2048['transient_fids'][()]
                y_2048 = data_2048['target_spectra'][()]
                ppm_2048 = data_2048["ppm"][()]
                t_2048 = data_2048["t"][()]

                x_4096 = data_4096['transient_fids'][()]
                y_4096 = data_4096['target_spectra'][()]
                ppm_4096= data_4096["ppm"][()]
                t_4096 = data_4096["t"][()]

                # visualize the 4096 data and the split
                if False:
                    import matplotlib.pyplot as plt
                    plt.figure()
                    x_4096 = np.fft.fftshift(np.fft.fft(x_4096[:, 1::2], axis=1), axes=1)
                    plt.plot(x_4096[0, :, 1, 0], 'k')
                    # plt.plot(t_4096[0], x_4096[0, :, 1])
                    plt.show()
                    stop


                if self.track == 'both':
                    x = np.concatenate((x, x_2048, x_4096[:, ::2], x_4096[:, 1::2]))
                    y = np.concatenate((y, y_2048, y_4096[:, ::2], y_4096[:, 1::2]))
                    ppm = np.concatenate((ppm, ppm_2048, ppm_4096[:, ::2], ppm_4096[:, 1::2]))
                    t = np.concatenate((t, t_2048, t_4096[:, ::2], t_4096[:, 1::2]))

                else:
                    x = np.concatenate((x_2048, x_4096[:, ::2], x_4096[:, 1::2]))
                    y = np.concatenate((y_2048, y_4096[:, ::2], y_4096[:, 1::2]))
                    ppm = np.concatenate((ppm_2048, ppm_4096[:, ::2], ppm_4096[:, 1::2]))
                    t = np.concatenate((t_2048, t_4096[:, ::2], t_4096[:, 1::2]))

        self.x, self.y, self.ppm, self.t = x, y, ppm, t

        # division of training and testing data
        x_train = x[:int(x.shape[0] * 0.9)]
        x_test = x[int(x.shape[0] * 0.9):]

        y_train = y[:int(y.shape[0] * 0.9)]
        y_test = y[int(y.shape[0] * 0.9):]

        ppm_train = ppm[:int(ppm.shape[0] * 0.9)]
        ppm_test = ppm[int(ppm.shape[0] * 0.9):]

        t_train = t[:int(t.shape[0] * 0.9)]
        t_test = t[int(t.shape[0] * 0.9):]

        # division of training and validation data
        x_val = x_train[int(x_train.shape[0] * 0.8):]
        x_train = x_train[:int(x_train.shape[0] * 0.8)]

        y_val = y_train[int(y_train.shape[0] * 0.8):]
        y_train = y_train[:int(y_train.shape[0] * 0.8)]

        ppm_val = ppm_train[int(ppm_train.shape[0] * 0.8):]
        ppm_train = ppm_train[:int(ppm_train.shape[0] * 0.8)]

        t_val = t_train[int(t_train.shape[0] * 0.8):]
        t_train = t_train[:int(t_train.shape[0] * 0.8)]

        if stage == 'train':
            self.x, self.y, self.ppm, self.t = x_train, y_train, ppm_train, t_train
        elif stage == 'val':
            self.x, self.y, self.ppm, self.t = x_val, y_val, ppm_val, t_val
        elif stage == 'test':
            self.x, self.y, self.ppm, self.t = x_test, y_test, ppm_test, t_test

    def __len__(self):
        return int(self.y.shape[0])

    def __getitem__(self, idx):
        x = self.x[idx].copy()

        if self.phase == 'test':
            # take first few transients for testing
            x = x[..., :self.transients]
        else:
            # draw transients randomly -> without multi-draw (i.e. replace=False)
            transIdxs = np.random.choice(x.shape[-1], size=self.transients, replace=False)
            x = x[..., transIdxs]

            # augmentation
            if self.transform is not None:
                for trans in self.transform:
                    x = trans(x, self.t[idx])

        # correct phase and frequency
        if self.register:
            x = spec_reg.basic_spectral_registration(x[np.newaxis, ...],
                                                     self.t[idx][np.newaxis, ...]).squeeze()
        # already in freq. domain
        y = self.y[idx].copy()
        y /= np.max(np.abs(y), axis=0, keepdims=True)

        # fft, subtraction, and normalization
        x, _ = self.processData(x, y)   # notice y is not used from here!

        if self.clean:
            x = y.copy()[..., np.newaxis]

        return x, y, self.ppm[idx], self.t[idx]


#**************************************************************************************************#
#                                      Class SubmissionDataset                                     #
#**************************************************************************************************#
#                                                                                                  #
# Dataset used to load data, process and iterate over the submission data set.                     #
#                                                                                                  #
#**************************************************************************************************#
class SubmissionDataset(MasterDataset):
    def __init__(self, data_dir, data_format='real', phase='train',
                 register=False, transients=40, type=None):
        self.type = type
        MasterDataset.__init__(self, data_dir, data_format, phase, register, transients)

    def setup(self, stage=None):
        with h5py.File(self.data_dir) as hf:
            if self.type == '2048':
                data = hf['data_2048']
            elif self.type == '4096':
                data = hf['data_4096']
            else:
                data = hf
            key = list(data.keys())
            x = data[key[2]][()]
            ppm = data[key[0]][()]
            t = data[key[1]][()]

        # downsample to 2048 if 4096
        if self.type == '4096':
            x = x[:, ::2]
            ppm = ppm[:, ::2]
            t = t[:, ::2]

        self.x, self.ppm, self.t = x, ppm, t

    def __len__(self):
        return int(self.x.shape[0])

    def __getitem__(self, idx):
        x = self.x[idx].copy()

        # take first few transients if less
        x = x[..., :self.transients]

        # correct phase and frequency
        if self.register:
            x = spec_reg.basic_spectral_registration(x[np.newaxis, ...],
                                                     self.t[idx][np.newaxis, ...]).squeeze()
        # fft, subtraction, and normalization
        x, _ = self.processData(x, x[..., 0])   # notice x passed for y!
        return x, self.ppm[idx], self.t[idx]