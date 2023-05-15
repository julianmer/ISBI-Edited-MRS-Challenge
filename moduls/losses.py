####################################################################################################
#                                           losses.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: S. Amirrajab (s.amirrajab@tue.nl)                                                       #
#          J. P. Merkofer (j.p.merkofer@tu.nl)                                                     #
#                                                                                                  #
# Created: 23/01/23                                                                                #
#                                                                                                  #
# Purpose: Definition of training losses.                                                          #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import torch
import torch.nn as nn

from challenge.metric_calculator import calculate_metrics, calculate_linewidth

#****************************************#
#   a helpful function to mask spectra   #
#****************************************#
def get_mask(ppm, lower, upper):
    mask = torch.ones_like(ppm, device=ppm.device)
    mask[ppm >= upper] = 0
    mask[ppm <= lower] = 0
    return mask


#**************************************************************************************************#
#                                         Class TotalLoss                                          #
#**************************************************************************************************#
#                                                                                                  #
# A loss based on the MAE, SNR, linewidth, and shape score.                                        #
#                                                                                                  #
#**************************************************************************************************#
class TotalLoss(nn.Module):
    def __init__(self):
        super(TotalLoss, self).__init__()

    def forward(self, x, y, ppm):
        # get all losses
        mse_loss = RangeMSELoss(noise=True, reduction='sum')(x, y, ppm)
        # snr_loss = SNRLoss()(x, y, ppm)
        linewidth_loss = LinewidthLoss()(x, y, ppm)
        shape_score_loss = ShapeLoss()(x, y, ppm)

        # print(linewidth_loss)
        # np_pred = x.float().cpu().detach().numpy()
        # np_ppm = ppm.float().cpu().detach().numpy()
        # model_metrics = calculate_linewidth(np_pred, np_ppm)
        # print(model_metrics)

        total = mse_loss + linewidth_loss + shape_score_loss
        return total


#**************************************************************************************************#
#                                         Class CustomLoss                                         #
#**************************************************************************************************#
#                                                                                                  #
# A loss based on the MAE, SNR, linewidth, and shape score.                                        #
#                                                                                                  #
#**************************************************************************************************#
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, x, y, ppm):
        mae_loss = RangeMSELoss(noise=True)(x, y, ppm)
        shape_score_loss = ShapeLoss()(x, y, ppm)

        total = mae_loss + shape_score_loss
        return total


#**************************************************************************************************#
#                                        Class RangeMAELoss                                        #
#**************************************************************************************************#
#                                                                                                  #
# A loss based on the MAE of spectra within a defined ppm range.                                   #
#                                                                                                  #
#**************************************************************************************************#
class RangeMAELoss(nn.Module):
    def __init__(self, noise=True, reduction='mean'):
        self.noise = noise
        self.reduction = reduction
        super(RangeMAELoss, self).__init__()

    def forward(self, x, y, ppm):
        # selecting part of arrays pertaining to region of interest
        loss_x = x * get_mask(ppm, 2.5, 4)
        loss_y = y * get_mask(ppm, 2.5, 4)

        if self.noise:   # selecting region of interest of noise and adding to loss
            loss_x += x * get_mask(ppm, 9.8, 10.8)
            loss_y += y * get_mask(ppm, 9.8, 10.8)

        # calculate absolute loss mean value
        return torch.nn.L1Loss(reduction=self.reduction)(loss_x, loss_y)


#**************************************************************************************************#
#                                        Class RangeMSELoss                                        #
#**************************************************************************************************#
#                                                                                                  #
# A loss based on the MSE of spectra within a defined ppm range.                                   #
#                                                                                                  #
#**************************************************************************************************#
class RangeMSELoss(nn.Module):
    def __init__(self, noise=True, reduction='mean'):
        self.noise = noise
        self.reduction = reduction
        super(RangeMSELoss, self).__init__()

    def forward(self, x, y, ppm):

        # selecting part of arrays pertaining to region of interest
        loss_x = x * get_mask(ppm, 2.5, 4)
        loss_y = y * get_mask(ppm, 2.5, 4)

        if self.noise:   # selecting region of interest of noise and adding to loss
            loss_x += x * get_mask(ppm, 9.8, 10.8)
            loss_y += y * get_mask(ppm, 9.8, 10.8)

        return torch.nn.MSELoss(reduction=self.reduction)(loss_x, loss_y)


#**************************************************************************************************#
#                                           Class SNRLoss                                          #
#**************************************************************************************************#
#                                                                                                  #
# A loss based on the SNR of GABA.                                                                 #
#                                                                                                  #
#**************************************************************************************************#
class SNRLoss(nn.Module):
    def __init__(self):
        super(SNRLoss, self).__init__()

    def forward(self, x, y, ppm):
        # get masks for signal and noise regions
        signal_mask = get_mask(ppm, 2.8, 3.2)
        noise_mask = get_mask(ppm, 9.8, 10.8)

        # calculate signal and noise
        signal = x * signal_mask
        noise = x * noise_mask

        # get max peak of signal and stdev of noise
        max_peak = torch.max(signal, -1).values
        stdevMan = torch.std(noise, -1)

        # calculate SNR
        snr = max_peak / 2 * stdevMan
        return torch.mean(1 / torch.log(snr + torch.finfo(torch.float32).eps))


#**************************************************************************************************#
#                                        Class LinewidthLoss                                       #
#**************************************************************************************************#
#                                                                                                  #
# A loss based on the linewidth of spectral region of GABA.                                        #
#                                                                                                  #
#**************************************************************************************************#
class LinewidthLoss(nn.Module):
    def __init__(self):
        super(LinewidthLoss, self).__init__()

    def forward(self, x, y, ppm):
        # computes a simplified version of the linewidth loss by mapping only the GABA region
        # larger than half the max peak of GABA to 1 as close as possible and then transforming
        # to the ppm scale

        # get GABA region
        gaba_mask = get_mask(ppm, 2.8, 3.2)
        gaba = x * gaba_mask
        gaba = (gaba - gaba.min()) / (gaba.max() - gaba.min())  # normalize

        # get max of GABA peaks
        max_peak = gaba.max(-1, keepdim=True).values

        # create mask for indices larger than half of max peak
        indices = torch.zeros_like(gaba, device=gaba.device)
        indices[gaba > 0.5 * max_peak] = 1.

        gaba = gaba * indices
        gaba = (gaba - gaba.min()) / (gaba.max() - gaba.min())  # normalize

        # use tanh to get a value between -1 and 1, were 1 is half the max peak
        gaba = torch.tanh(torch.pi / 2 * gaba)   # by scaling the input we can get a more
        gaba = gaba * indices   # turn -1 to 0   # accurate approximation of the linewidth

        # print(indices.sum(-1), gaba.sum(-1))

        # # calculate linewidth
        # linewidth = indices.sum(-1) * torch.abs((ppm[:, 0] - ppm[:, 1]))

        # to get a gradient we return a sum of the gaba peaks above half the max peak
        return torch.mean(gaba.sum(-1) * torch.abs((ppm[:, 0] - ppm[:, 1])))


#**************************************************************************************************#
#                                          Class ShapeLoss                                         #
#**************************************************************************************************#
#                                                                                                  #
# A loss based on the shape of the spectral region of GABA and Glx.                                #
#                                                                                                  #
#**************************************************************************************************#
class ShapeLoss(nn.Module):
    def __init__(self):
        super(ShapeLoss, self).__init__()

    def forward(self, x, y, ppm):
        gaba_mask = get_mask(ppm, 2.8, 3.2)
        glx_mask = get_mask(ppm, 3.6, 3.9)

        # get GABA and Glx regions
        gaba = x * gaba_mask
        glx = x * glx_mask
        gaba_gt = y * gaba_mask
        glx_gt = y * glx_mask

        # normalize
        gaba = (gaba - gaba.min()) / (gaba.max() - gaba.min())
        glx = (glx - glx.min()) / (glx.max() - glx.min())
        gaba_gt = (gaba_gt - gaba_gt.min()) / (gaba_gt.max() - gaba_gt.min())
        glx_gt = (glx_gt - glx_gt.min()) / (glx_gt.max() - glx_gt.min())

        # calculate shape correlation
        gaba_corr = torch.nn.functional.cosine_similarity(gaba, gaba_gt, dim=-1)
        glx_corr = torch.nn.functional.cosine_similarity(glx, glx_gt, dim=-1)

        return torch.mean(1 - (0.6 * gaba_corr + 0.4 * glx_corr))


#**************************************************************************************************#
#                                          Class ExpLoss                                           #
#**************************************************************************************************#
#                                                                                                  #
# A loss based on the exponential rangeMAE, MSE, ...                                               #
#                                                                                                  #
#**************************************************************************************************#
class ExpLoss(nn.Module):
    def __init__(self):
        super(ExpLoss, self).__init__()

    def forward(self, x, y, ppm):
        mse = RangeMSELoss(noise=True)(x, y, ppm)
        return torch.exp(mse) - 1

#**************************************************************************************************#
#                                          Class ExpLoss                                           #
#**************************************************************************************************#
#                                                                                                  #
# A loss based on the exponential rangeMAE, MSE, ...                                               #
#                                                                                                  #
#**************************************************************************************************#
class ExpLoss(nn.Module):
    def __init__(self):
        super(ExpLoss, self).__init__()

    def forward(self, x, y, ppm):
        mse = MSELoss()(x, y, ppm)
        return torch.exp(mse) - 1


#**************************************************************************************************#
#                                           Class MAELoss                                          #
#**************************************************************************************************#
#                                                                                                  #
# A loss based on the MAE of spectra.                                                              #
#                                                                                                  #
#**************************************************************************************************#
class MAELoss(nn.Module):
    def __init__(self):
        super(MAELoss, self).__init__()

    def forward(self, x, y, ppm):
        return torch.nn.L1Loss()(x, y)


#**************************************************************************************************#
#                                           Class MSELoss                                          #
#**************************************************************************************************#
#                                                                                                  #
# A loss based on the MAE of spectra.                                                              #
#                                                                                                  #
#**************************************************************************************************#
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, x, y, ppm):
        return torch.nn.MSELoss()(x, y.float())