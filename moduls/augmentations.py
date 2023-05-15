####################################################################################################
#                                         augmentations.py                                         #
####################################################################################################
#                                                                                                  #
# Authors: S. Amirrajab (s.amirrajab@tue.nl)                                                       #
#                                                                                                  #
# Created: 23/01/23                                                                                #
#                                                                                                  #
# Purpose: Definition of augmentations for edited MRS data.                                        #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import h5py
import math
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

# own
from challenge import spec_reg



#**************************************************************************************************#
#                                    Class BaselineAugmentaion                                     #
#**************************************************************************************************#
#                                                                                                  #
# Adds amplitude, phasek, and frequency noise to create transients.                                #
#                                                                                                  #
#**************************************************************************************************#
class BaselineAugmentaion:
    """
    Adds frequency shift to scans according to a normal distribution.

    Parameters:
        transients (int): The number of transients to be generated.
    """
    def __init__(self, transients=40):
        self.transients = transients

    def __call__(self, fids, t):
        """
        Apply the frequency shift transformation to the given fids.

        Parameters:
            fids (np.ndarray): An array of FIDs.
            t (np.ndarray): An array of time values corresponding to the FIDs.

        Returns:
            tuple: The shifted FIDs and the original (unshifted) FIDs.
        """
        if len(fids.shape) < 3:
            noise_fid = np.expand_dims(fids, axis=2)
            noise_fid = np.repeat(noise_fid, self.transients, axis=2)
        else:
            noise_fid = fids

        # amplitude noise
        base_amplitude_noise = 10
        amplitude_noise = np.random.normal(0, base_amplitude_noise, size=noise_fid.shape)
        noise_fid = noise_fid + amplitude_noise

        # frequency noise
        freq_shift_std = 20
        freq_shifts = np.random.normal(0, freq_shift_std, size=(1, 1, noise_fid.shape[2]))
        noise_fid = noise_fid * np.exp(1j * (freq_shifts * t.reshape(t.shape[0], 1, 1) * math.pi / 2))

        # phase noise
        phase_shift_std = 30
        phase_shifts = np.random.normal(0, phase_shift_std, size=(1, 1, noise_fid.shape[2]))
        noise_fid = noise_fid * np.exp(1j * (phase_shifts * np.ones((t.shape[0], 1, 1)) * math.pi / 180))

        return noise_fid


#**************************************************************************************************#
#                                    Class RandomFrequencyShift                                    #
#**************************************************************************************************#
#                                                                                                  #
# Adds frequency shift to scans according to a normal distribution.                                #
#                                                                                                  #
#**************************************************************************************************#
class RandomFrequencyShift:
    """
    Adds frequency shift to scans according to a normal distribution.

    Parameters:
        mean (float, Hz): The mean value of the normal distribution.
        std (float, Hz): The variance of the normal distribution.
        transients (int): The number of transients to be generated.
    """
    def __init__(self, mean=7, std=3, transients=40):
        self.mean = mean
        self.std = std
        self.transients = transients

    def __call__(self, fids, t):
        """
        Apply the frequency shift transformation to the given fids.

        Parameters:
            fids (np.ndarray): An array of FIDs.
            t (np.ndarray): An array of time values corresponding to the FIDs.

        Returns:
            tuple: The shifted FIDs and the original (unshifted) FIDs.
        """
        # Perform the frequency shift transformation.
        # ...

        if len(fids.shape) < 3:
            self.fids = np.expand_dims(fids, axis=2)
            self.fids = np.repeat(self.fids, self.transients, axis=2)
        else:
            self.fids = fids
        self.t = t

        plus_minus = (-1) ** (np.random.randint(1, 3))
        freq_noise = plus_minus * np.random.normal(self.mean, self.std, size=(1, 1, self.fids.shape[-1]))

        # noise = np.random.normal(0,base_noise.reshape(-1,1,1),size=(self.fids.shape[0],1,self.fids.shape[2]))
        # freq_noise = np.random.uniform(-base_noise.reshape(-1,1,1),base_noise.reshape(-1,1,1),size=self.fids.shape)
        # freq_noise = np.repeat(freq_noise,self.fids.shape[1], axis=1 )
        freq_noise = np.repeat(freq_noise, self.fids.shape[0], axis=0)
        freq_noise = np.repeat(freq_noise, self.fids.shape[1], axis=1)

        time = self.t.reshape(self.t.shape[0], 1, 1)
        time = np.repeat(time, self.fids.shape[1], axis=1)
        time = np.repeat(time, self.fids.shape[-1], axis=-1)
        self.fids = self.fids * np.exp(1j * time * freq_noise * 2 * np.pi)

        return self.fids


#**************************************************************************************************#
#                                      Class RandomPhaseShift                                      #
#**************************************************************************************************#
#                                                                                                  #
# Adds phase shift to scans according to a normal distribution.                                    #
#                                                                                                  #
#**************************************************************************************************#
class RandomPhaseShift:
    """
    Adds shift shift to scans according to a normal distribution.

    Parameters:
        mean (float, radians): The mean value of the normal distribution.
        std (float, radians): The variance of the normal distribution.
        transients (int, optional): The number of transients to be generated.
    """
    def __init__(self, mean=3.14, std=3, transients=40):
        self.mean = mean
        self.std = std
        self.transients = transients

    def __call__(self, fids, t):
        """
        Apply the shift shift transformation to the given fids.

        Parameters:
            fids (np.ndarray): An array of FIDs.
            t (np.ndarray): An array of time values corresponding to the FIDs.

        Returns:
            tuple: The shifted FIDs and the original (unshifted) FIDs.
        """
        # Perform the frequency shift transformation.
        # ...

        if len(fids.shape) < 3:
            self.fids = np.expand_dims(fids, axis=2)
            self.fids = np.repeat(self.fids, self.transients, axis=2)
        else:
            self.fids = fids
        self.t = t

        plus_minus = (-1) ** (np.random.randint(1, 3))
        phase_noise = plus_minus * np.random.normal(self.mean, self.std, size=(1, 1, self.fids.shape[-1]))
        phase_noise = np.repeat(phase_noise, self.fids.shape[0], axis=0)
        phase_noise = np.repeat(phase_noise, self.fids.shape[1], axis=1)

        self.fids = self.fids * np.exp(1j * phase_noise)

        return self.fids


#**************************************************************************************************#
#                                        Class RandomNoise                                         #
#**************************************************************************************************#
#                                                                                                  #
# Adds normal amplitude noise to time-domain data according to a normal distribution.              #
#                                                                                                  #
#**************************************************************************************************#
class RandomNoise:
    def __init__(self, mean=10, std=3, transients=40):
        """
        Initialize the class with the following parameters:
        :param mean: Mean value applied to all transients and scans (float)
        :param var: Level of variation between different scans (float)
        :param transients: Number of transients to add noise to (int)
        """
        self.mean = mean
        self.std = std
        self.transients = transients

    def __call__(self, fids, t):
        """
        Add normal amplitude noise to time-domain data
        :param fids: Array of FIDs (np.ndarray)
        :param time_points: Array of time points corresponding to FIDs (np.ndarray)
        :return: Tuple of noisy FIDs and ground truth FIDs (np.ndarray, np.ndarray)
        """
        # Check if input data is of the correct shape and data type
        # if not isinstance(fids, np.ndarray) or not isinstance(time_points, np.ndarray):
        #     raise TypeError("Input data must be of type numpy.ndarray")
        # if fids.ndim != 1 or time_points.ndim != 1:
        #     raise ValueError("Input data must be 1-dimensional arrays")
        # if fids.shape[0] != time_points.shape[0]:
        #     raise ValueError("FIDs and time_points must have the same length")

        if len(fids.shape) < 3:
            self.fids = np.expand_dims(fids, axis=2)
            self.fids = np.repeat(self.fids, self.transients, axis=2)
        else:
            self.fids = fids
        self.t = t

        # base_noise = self.mean * np.ones(self.fids.shape[0]) + np.random.uniform(low=-self.std, high=self.std, size=self.fids.shape[0])

        # Add real and imaginary noise
        noise_real = ((-1) ** (np.random.randint(1, 3))) * np.random.normal(self.mean, self.std, size=self.fids.shape)
        noise_imag = ((-1) ** (np.random.randint(1, 3))) * 1j * np.random.normal(self.mean, self.std,
                                                                                 size=self.fids.shape)
        noise_complex = noise_real + noise_imag
        self.fids = self.fids + noise_complex
        return self.fids


#**************************************************************************************************#
#                                      Class RandomLineBroad                                       #
#**************************************************************************************************#
#                                                                                                  #
# Adds line broadening to scans according to a normal distribution.                                #
#                                                                                                  #
#**************************************************************************************************#
class RandomLineBroad:
    def __init__(self, mean=10, std=3, transients=40, filter='exp'):
        """
        Initialize the class with the following parameters:
        :param mean: broadening (tuple,float): apodisation in Hz
        :param var: Level of variation between different scans (float)
        :param transients: Number of transients to add noise to (int)
        :param filter (str,optional):'exp','l2g'
        """
        self.mean = mean
        self.std = std
        self.transients = transients
        self.filter = filter

    def __call__(self, fids, t):
        """ Apodize FID
        Args:
            FID (ndarray): Time domain data
            dwelltime (float): dwelltime in seconds
            broadening (tuple,float): apodisation in Hz
            filter (str,optional):'exp','l2g'
        Returns:
            FID (ndarray): Apodised FID
        """
        if len(fids.shape) < 3:
            self.fids = np.expand_dims(fids, axis=2)
            self.fids = np.repeat(self.fids, self.transients, axis=2)
        else:
            self.fids = fids
        self.t = t

        time = self.t.reshape(self.t.shape[0], 1, 1)
        time = np.repeat(time, self.fids.shape[1], axis=1)
        time = np.repeat(time, self.fids.shape[-1], axis=-1)
        if self.filter == 'exp':
            broadening = np.random.normal(self.mean, self.std, size=(1, 1, self.fids.shape[-1]))
            broadening = np.repeat(broadening, self.fids.shape[0], axis=0)
            broadening = np.repeat(broadening, self.fids.shape[1], axis=1)
            window = np.exp(-time * broadening[0])
        elif self.filter == 'l2g':
            broadeningl = np.random.normal(self.mean, self.std, size=(1, 1, self.fids.shape[-1]))
            broadeningl = np.repeat(broadeningl, self.fids.shape[0], axis=0)
            broadeningl = np.repeat(broadeningl, self.fids.shape[1], axis=1)

            broadeningb = np.random.normal(self.mean, self.std, size=(1, 1, self.fids.shape[-1]))
            broadeningb = np.repeat(broadeningb, self.fids.shape[0], axis=0)
            broadeningb = np.repeat(broadeningb, self.fids.shape[1], axis=1)

            # make sure they are positive
            broadeningl, broadeningb = np.sqrt(broadeningl ** 2), np.sqrt(broadeningb ** 2)

            window = np.exp(-time * (broadeningl + time * broadeningb ** 2))
        else:
            print('Filter not recognised, should be "exp" or "l2g".')
            window = 1

        self.fids = window * self.fids
        return self.fids


#**************************************************************************************************#
#                                         Class RandomPeak                                         #
#**************************************************************************************************#
#                                                                                                  #
# Adds one random peak at the specified ppm, only for ON signal (same for all transients).         #
#                                                                                                  #
#**************************************************************************************************#
class RandomPeak:
    def __init__(self, ppm=(3, 3.3), amp=(0.1, 5), gamma=(5, 25), sigma=(5, 25), transients=40):
        """
        Initialize the class with the following parameters:

        :param ppm: (min, max) of a uniform distribution
        :param amp: (min, max) of a uniform distribution
        :param gamma: (min, max) of a uniform distribution Lorentzian line broadening
        :param sigma: (min, max) of a uniform distribution Gaussian line broadening
        :param transients: Number of transients to add a random peak to (int)

        Function creates one random peak at the specified ppm, only for On signal
        same for all transients
        """
        self.ppm_range = ppm
        self.amp_range = amp

        self.gamma = gamma
        self.sigma = sigma
        self.transients = transients
        # H2O_PPM_TO_TMS = 4.65

    def create_peak(self, time_axis):
        """
            creates FID for peak at specific ppm
        Parameters
        ----------
        gamma     : Lorentzian line broadening
        sigma     : Gaussian line broadening
        Returns
        -------
        array-like FID (2048, 2, # of transients)
        """
        out = np.zeros(time_axis.shape, dtype=np.complex128)

        for p, a in zip(self.ppm, self.amp):
            gamma1 = np.random.uniform(self.gamma[0], self.gamma[1], 1)
            sigma1 = np.random.uniform(self.sigma[0], self.sigma[1], 1)

            ppm_on = 128 * (p - 3)  # ppm to Hz
            T2s = 0.05
            x = a * np.exp(1j * 2 * np.pi * ppm_on * time_axis) * np.exp(-time_axis / T2s)

            if gamma1 > 0 or sigma1 > 0:
                x = self.blur_FID_Voigt(time_axis, x, gamma1, sigma1)

            out += x

        return out

    def blur_FID_Voigt(self, time_axis, FID, gamma, sigma):
        """
        Blur FID in spectral domain
        Parameters:
        time_axis : time_axis
        FID       : array-like
        gamma     : Lorentzian line broadening
        sigma     : Gaussian line broadening
        Returns:
        array-like
        """
        FID_blurred = FID * np.exp(-time_axis * (gamma + time_axis * sigma ** 2 / 2))
        return FID_blurred

    def __call__(self, fids, t):
        """Add a peak.

        :param gamma: Lorentzian broadening, defaults to 0
        :type gamma: float, optional
        :param sigma: Gaussian broadening, defaults to 0

        """
        self.ppm = np.random.uniform(self.ppm_range[0], self.ppm_range[1], 1)
        self.amp = np.random.uniform(self.amp_range[0], self.amp_range[1], 1)
        self.t = t
        if len(fids.shape) < 3:
            self.fids = np.expand_dims(fids, axis=2)
            self.fids = np.repeat(self.fids, self.transients, axis=2)

        else:
            self.fids = fids
        time = np.expand_dims(self.t[:, np.newaxis], axis=2)
        time = np.repeat(time, self.transients, axis=2)

        fid_on = self.create_peak(time)
        fid_off = np.zeros_like(fid_on)
        fid = np.concatenate((fid_off, fid_on), axis=1)

        self.fids = fid + self.fids
        return self.fids


#**************************************************************************************************#
#                                        Class RandomMMPeak                                        #
#**************************************************************************************************#
#                                                                                                  #
# Adds peaks at specified ppm locations, only for ON signal (same for all transients).             #
#                                                                                                  #
#**************************************************************************************************#
class RandomMMPeak:
    def __init__(self, ppm=[0.93, 1.24, 1.43, 1.72, 2.05, 2.29, 3.0, 3.20, 3.8, 4.3],
                 amp=(0.1, 5), gamma=(5, 25), sigma=(5, 25), transients=40):
        """
        Initialize the class with the following parameters:

        :param ppm: ppm values for MM signal (M1 to M10 from de Graaf book page 74)
        :param amp: (min, max) of a uniform distribution
        :param gamma: (min, max) of a uniform distribution Lorentzian line broadening
        :param sigma: (min, max) of a uniform distribution Gaussian line broadening
        :param transients: Number of transients to add a random peak to (int)

        Function creates one random peak at the specified ppm, only for On signal
        same for all transients
        """
        self.ppm = [ppm] if not isinstance(ppm, list) else ppm
        self.amp = [np.random.uniform(amp[0], amp[1]) for i in range(len(self.ppm))] \
            if not isinstance(amp, list) else np.random.uniform(
            amp[0], amp[1])
        self.transients = transients
        self.gamma = gamma
        self.sigma = sigma
        assert len(self.ppm) == len(self.amp)
        # H2O_PPM_TO_TMS = 4.65

    def create_peak(self, time_axis):
        """
            creates FID for peak at specific ppm
        Parameters
        ----------
        gamma     : Lorentzian line broadening
        sigma     : Gaussian line broadening
        Returns
        -------
        array-like FID (2048, 2, # of transients)
        """

        out = np.zeros(time_axis.shape, dtype=np.complex128)

        for p, a in zip(self.ppm, self.amp):
            gamma1 = np.random.uniform(self.gamma[0], self.gamma[1], 1)
            sigma1 = np.random.uniform(self.sigma[0], self.sigma[1], 1)
            ppm_on = 128 * (p - 3)  # ppm to Hz
            T2s = 0.05
            x = a * np.exp(1j * 2 * np.pi * ppm_on * time_axis) * np.exp(-time_axis / T2s)

            if gamma1 > 0 or sigma1 > 0:
                x = self.blur_FID_Voigt(time_axis, x, gamma1, sigma1)

            out += x
        return out

    def blur_FID_Voigt(self, time_axis, FID, gamma, sigma):
        """
        Blur FID in spectral domain
        Parameters:
        time_axis : time_axis
        FID       : array-like
        gamma     : Lorentzian line broadening
        sigma     : Gaussian line broadening
        Returns:
        array-like
        """
        FID_blurred = FID * np.exp(-time_axis * (gamma + time_axis * sigma ** 2 / 2))
        return FID_blurred

    def __call__(self, fids, t):
        """Add a MM peak at
        ppm=[0.93, 1.24, 1.43, 1.72, 2.05, 2.29, 3.0, 3.20, 3.8, 4.3]
        """
        self.t = t
        if len(fids.shape) < 3:
            self.fids = np.expand_dims(fids, axis=2)
            self.fids = np.repeat(self.fids, self.transients, axis=2)

        else:
            self.fids = fids
        time = np.expand_dims(self.t[:, np.newaxis], axis=2)
        time = np.repeat(time, self.transients, axis=2)

        fid_on = self.create_peak(time)
        fid_off = np.zeros_like(fid_on)
        fid = np.concatenate((fid_off, fid_on), axis=1)

        self.fids = fid + self.fids
        return self.fids
