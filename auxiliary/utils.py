####################################################################################################
#                                             utils.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: S. Amirrajab (s.amirrajab@tue.nl),                                                      #
#          J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 23/01/23                                                                                #
#                                                                                                  #
# Purpose: Some helpful functions are defined here.                                                #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb


#******************************#
#   define metrics for wandb   #
#******************************#
def define_metrics():
    wandb.define_metric("mse", summary="mean")
    wandb.define_metric("snr", summary="mean")
    wandb.define_metric("linewidth", summary="mean")
    wandb.define_metric("shape_score", summary="mean")

    wandb.define_metric("in-vivo mse", summary="mean")
    wandb.define_metric("in-vivo snr", summary="mean")
    wandb.define_metric("in-vivo linewidth", summary="mean")
    wandb.define_metric("in-vivo shape_score", summary="mean")

    wandb.define_metric("cha-sims mse", summary="mean")
    wandb.define_metric("cha-sims snr", summary="mean")
    wandb.define_metric("cha-sims linewidth", summary="mean")
    wandb.define_metric("cha-sims shape_score", summary="mean")

    wandb.define_metric("sub-track1 snr", summary="mean")
    wandb.define_metric("sub-track1 linewidth", summary="mean")

    wandb.define_metric("sub-track2 snr", summary="mean")
    wandb.define_metric("sub-track2 linewidth", summary="mean")


    wandb.define_metric("sub-track3 snr", summary="mean")
    wandb.define_metric("sub-track3 linewidth", summary="mean")


#****************************#
#   log train and val loss   #
#****************************#
def log_loss_wandb(train_loss, val_loss):
    # log metrics to visualize performance
    wandb.log({"train_loss": train_loss.item(), "val_loss": val_loss.item()})


#***********************#
#   log plot of specs   #
#***********************#
def plot_spectra(data_loader, model, device, epoch=0, show=False):
    data = next(iter(data_loader))
    y = data['y']
    x = data['x'].to(device)
    ppm_axis = data['ppm'].to(device)

    spec_rec = model(x)

    if x.shape[0] >= 5: batches = [0, 2, 4]
    else: batches = list(range(x.shape[0]))

    y = y.detach().cpu()
    spec_rec = spec_rec.detach().cpu()
    ppm_axis = ppm_axis.detach().cpu()
    fig, axs = plt.subplots(1, len(batches), figsize=(15,5))
    axs = axs.ravel()
    for i in range(len(batches)):
            axs[i].plot(ppm_axis[batches[i],:], y[batches[i], :], alpha=0.5, label="GT")
            axs[i].plot(ppm_axis[batches[i],:], spec_rec[batches[i], :], label="Recon")
            axs[i].set_xlim(1, 5)
            axs[i].invert_xaxis()
            axs[i].grid()
            axs[i].legend()

    plt.tight_layout()
    if show:
        plt.show()
    wandb.log({f"GT vs. Recon, epoch {epoch + 1}": fig})


#**********************************#
#   log plot of specs from batch   #
#**********************************#
def plot_spectra_batch(batch, model, show=False):
    x, y, ppm_axis, t = batch

    if x.shape[0] >= 5: batches = [0, 2, 4]
    else: batches = list(range(x.shape[0]))

    spec_rec = model(x, ppm_axis, t)

    y = y.float().detach().cpu()
    spec_rec = spec_rec.float().detach().cpu()
    ppm_axis = ppm_axis.float().detach().cpu()
    fig, axs = plt.subplots(1, len(batches), figsize=(len(batches) * 5, 5))
    axs = axs.ravel()
    for i in range(len(batches)):
        axs[i].plot(ppm_axis[batches[i], :], y[batches[i], :], alpha=0.5, label="GT")
        axs[i].plot(ppm_axis[batches[i], :], spec_rec[batches[i], :], label="Recon")
        axs[i].set_xlim(1, 5)
        axs[i].invert_xaxis()
        axs[i].grid()
        axs[i].legend()

    plt.tight_layout()
    if show:
        plt.show()
    return fig


#*************************************#
#   log plot of spectrum from batch   #
#*************************************#
def plot_spectrum(pred, ppm, show=False):
    if pred.shape[0] >= 5: batches = [0, 2, 4]
    else: batches = list(range(pred.shape[0]))

    fig, axs = plt.subplots(1, len(batches), figsize=(len(batches) * 5, 5))
    axs = axs.ravel()
    for i in range(len(batches)):
        axs[i].plot(ppm[batches[i], :], pred[batches[i], :], label="Recon", color='tab:orange')
        axs[i].set_xlim(1, 5)
        axs[i].invert_xaxis()
        axs[i].grid()
        axs[i].legend()

    plt.tight_layout()
    if show:
        plt.show()
    return fig


#*********************************#
#   log plot of covs from batch   #
#*********************************#
def plot_covs_batch(batch, show=False):
    x, y, ppm_axis, t = batch

    if x.shape[0] >= 5: batches = [0, 2, 4]
    else: batches = list(range(x.shape[0]))

    x = x.float()
    if len(x.shape) > 3: x = x.squeeze()
    cov = torch.matmul(x, torch.permute(x, (0, 2, 1)))
    cov = cov.unsqueeze(1).cpu()
    # cov = torch.nn.BatchNorm2d(num_features=1)(cov.float())
    cov = torch.nn.functional.normalize(cov.float())
    cov = cov.squeeze().detach().numpy()

    figX, axs = plt.subplots(1, len(batches), figsize=(len(batches) * 5, 5))
    axs = axs.ravel()
    for i in range(len(batches)):
        axs[i].imshow(cov[batches[i]], cmap='gray')

    y = y.float().unsqueeze(-1)
    cov = torch.matmul(y, torch.permute(y, (0, 2, 1)))
    cov = cov.unsqueeze(1).cpu()
    # cov = torch.nn.BatchNorm2d(num_features=1)(cov.float())
    cov = torch.nn.functional.normalize(cov.float())
    cov = cov.squeeze().detach().numpy()

    figY, axs = plt.subplots(1, len(batches), figsize=(len(batches) * 5, 5))
    axs = axs.ravel()
    for i in range(len(batches)):
        axs[i].imshow(cov[batches[i]], cmap='gray')

    plt.tight_layout()
    if show:
        plt.show()
    return figX, figY


#****************************************************#
#   compare the spectra with and without the noise   #
#****************************************************#
def plot_edited(diff_spec, noise_diff_spec, ppm, show=False):
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))

    ax[0].plot(ppm[0], np.real(diff_spec[0]))
    ax[0].set_title("Ground Truth Difference Spectrum")

    ax[1].plot(ppm[0], np.real(noise_diff_spec.mean(axis=1)))
    ax[1].set_title("Reconstruction Average")

    for i in range(noise_diff_spec.shape[1]):
        ax[2].plot(ppm[0], np.real(noise_diff_spec[:, i]), alpha=0.3)
    ax[2].set_title("Reconstructed Transients")

    for i in range(3):
        ax[i].set_xlim(1.5,4)
        ax[i].invert_xaxis()
        ax[i].set_ylim(-1,1)
        ax[i].set_yticks([])

    plt.tight_layout()
    if show:
        plt.show()
    return fig