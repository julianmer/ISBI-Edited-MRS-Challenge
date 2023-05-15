####################################################################################################
#                                           sigModels.py                                           #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 07/07/22                                                                                #
#                                                                                                  #
# Purpose: Definitions of various MRS signal models.                                               #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import numpy as np
import torch


#**************************************************************************************************#
#                                          Class SigModel                                          #
#**************************************************************************************************#
#                                                                                                  #
# The base class for the MRS signal models. Defines the necessary attributes and methods a signal  #
# model should implement.                                                                          #
#                                                                                                  #
#**************************************************************************************************#
class SigModel():

    #*************************#
    #   initialize instance   #
    #*************************#
    def __init__(self, basis, baseline, order, first, last, t, f):
        self.basis = basis
        self.first, self.last = int(first), int(last)

        if not baseline:
            self.baseline = \
            torch.from_numpy(self.baseline_init(order, self.first, self.last))
        else:
            self.baseline = baseline

        self.t = t
        self.f = f

    #********************#
    #   parameter init   #
    #********************#
    def initParam(self, specs):
        pass

    #*******************#
    #   forward model   #
    #*******************#
    def forward(self, theta):
        pass

    #***************#
    #   regressor   #
    #***************#
    def regress_out(self, x, conf, keep_mean=True):
        """
        Linear deconfounding

        Ref: Clarke WT, Stagg CJ, Jbabdi S. FSL-MRS: An end-to-end spectroscopy analysis package.
        Magnetic Resonance in Medicine 2021;85:2950–2964 doi: https://doi.org/10.1002/mrm.28630.
        """
        if isinstance(conf, list):
            confa = np.squeeze(np.asarray(conf)).T
        else:
            confa = conf
        if keep_mean:
            m = np.mean(x, axis=0)
        else:
            m = 0
        return x - confa @ (np.linalg.pinv(confa) @ x) + m

    #*******************#
    #   baseline init   #
    #*******************#
    def baseline_init(self, order, first, last):
        x = np.zeros(self.basis.shape[0], complex)
        x[first:last] = np.linspace(-1, 1, last - first)
        B = []
        for i in range(order + 1):
            regressor = x ** i
            if i > 0:
                regressor = self.regress_out(regressor, B, keep_mean=False)

            B.append(regressor.flatten())
            B.append(1j * regressor.flatten())

        B = np.asarray(B).T
        tmp = B.copy()
        B = 0 * B
        B[first:last, :] = tmp[first:last, :].copy()
        return B



#**************************************************************************************************#
#                                         Class VoigtModel                                         #
#**************************************************************************************************#
#                                                                                                  #
# Implements a signal model as mentioned in [1] (a Voigt signal model).                            #
#                                                                                                  #
# [1] Clarke, W.T., Stagg, C.J., and Jbabdi, S. (2020). FSL-MRS: An end-to-end spectroscopy        #
#     analysis package. Magnetic Resonance in Medicine, 85, 2950 - 2964.                           #
#                                                                                                  #
#**************************************************************************************************#
class VoigtModel(SigModel):

    #*************************#
    #   initialize instance   #
    #*************************#
    def __init__(self, basis, first, last, t, f, baseline=None, order=2):
        """
        Main init for the VoigtModel class.

        @param basis -- The basis set of metabolites.
        @param baseline -- The baseline used to fit the spectra.
        """
        super(VoigtModel, self).__init__(basis, baseline, order,
                                         first=first, last=last, t=t, f=f)
    #********************#
    #   parameter init   #
    #********************#
    def initParam(self, specs):
        """
        Initializes the optimization parameters.

        @param specs -- The batch of specs to get initializations for.

        @returns -- The optimization parameters
        """
        theta = np.zeros((specs.shape[0], 41))
        theta[:, :self.basis.shape[1]] = np.random.rand(specs.shape[0], self.basis.shape[1])
        return torch.Tensor(theta)

    #*******************#
    #   forward model   #
    #*******************#
    def forward(self, theta):
        """
        The (forward) signal model.

        @returns -- The forward model function.
        """
        n = self.basis.shape[1]
        g = 1

        con = theta[:, :n]  # concentrations
        gamma = theta[:, n:n + g]  # lorentzian blurring
        sigma = theta[:, n + g:n + 2 * g]  # gaussian broadening
        eps = theta[:, n + 2 * g:n + 3 * g]  # frequency shift
        phi0 = theta[:, n + 3 * g]  # global phase shift
        phi1 = theta[:, n + 3 * g + 1]  # global phase ramp
        b = theta[:, n + 3 * g + 2:]  # baseline params

        # compute m(t) * exp(- (1j * eps + gamma + sigma ** 2 * t) * t)
        lin = torch.exp(- (1j * eps + gamma + (sigma ** 2) * self.t) * self.t)
        ls = lin[..., None] * self.basis

        S = torch.fft.fft(ls, dim=1)

        # compute exp(-1j * (phi0 + phi1 * nu)) * con * S(nu)
        ex = torch.exp(-1j * (phi0[..., None] + phi1[..., None] * self.f))
        fd = ex * torch.sum(con[:, None, :] * S, -1)

        # add baseline
        if self.baseline is not None:

            # compute baseline
            if len(self.baseline.shape) > 2:
                ba = torch.einsum("ij, ikj -> ik", b.cfloat(), self.baseline)
            else:
                ba = torch.einsum("ij, kj -> ik", b.cdouble(), self.baseline)

            fd += ba

        return fd

    #*********************#
    #   gradient vector   #
    #*********************#
    def gradient(self, theta, specs, constr=False):
        """
        The gradient of the signal model.

        @returns -- The gradient.
        """
        n = self.basis.shape[1]
        g = 1

        # # ! make sure specs are processed, otherwise call: processSpec(specs)
        # specs = specs[:, 0] + 1j * specs[:, 1]

        con = theta[:, :n]  # concentrations
        gamma = theta[:, n:n + g]  # lorentzian blurring
        sigma = theta[:, n + g:n + 2 * g]  # gaussian broadening
        eps = theta[:, n + 2 * g:n + 3 * g]  # frequency shift
        phi0 = theta[:, n + 3 * g]  # global phase shift
        phi1 = theta[:, n + 3 * g + 1]  # global phase ramp
        b = theta[:, n + 3 * g + 2:]  # baseline params

        # compute m(t) * exp(- (1j * eps + gamma + sigma ** 2 * t) * t)
        lin = torch.exp(- (1j * eps + gamma + (sigma ** 2) * self.t) * self.t)
        ls = lin[..., None] * self.basis

        S = torch.fft.fft(ls, dim=1)

        # compute exp(-1j * (phi0 + phi1 * nu))
        ex = torch.exp(-1j * (phi0[..., None] + phi1[..., None] * self.f))

        fd = ex * torch.sum(con[:, None, :] * S, -1)

        ea = ex[..., None] * con[:, None, :]

        Sg = torch.fft.fft(- self.t[None, :, None] * ls, dim=1)
        Ss = torch.fft.fft(- 2 * sigma[..., None] * self.t[None, :, None] ** 2 * ls, dim=1)
        Se = torch.fft.fft(- 1j * self.t[None, :, None] * ls, dim=1)

        dc = ex[..., None] * S
        dg = torch.sum(ea * Sg, -1)
        ds = torch.sum(ea * Ss, -1)
        de = torch.sum(ea * Se, -1)
        dp0 = - 1j * torch.sum(ea * S, -1)
        dp1 = - 1j * self.f * torch.sum(ea * S, -1)

        if not len(self.baseline.shape) > 2:
            fd += torch.einsum("ij, kj -> ik", b.cdouble(), self.baseline)
            db = self.baseline.repeat((specs.shape[0], 1, 1))
        elif type(self.baseline) is torch.Tensor:
            fd += torch.einsum("ij, ikj -> ik", b.cfloat(), self.baseline)
            db = self.baseline

        dS = torch.cat((dc, dg.unsqueeze(-1), ds.unsqueeze(-1), de.unsqueeze(-1),
                        dp0.unsqueeze(-1), dp1.unsqueeze(-1), db), dim=-1)

        grad = torch.real((fd[..., None] * torch.conj(dS) + torch.conj(fd)[..., None] * dS -
                           torch.conj(specs)[..., None] * dS - specs[..., None] * torch.conj(dS)))

        return grad.sum(1)
