# A DEEP LEARNING APPROACH UTILIZING COVARIANCE MATRIX ANALYSIS FOR THE ISBI EDITED MRS RECONSTRUCTION CHALLENGE

## Abstract

Edited magnetic resonance spectroscopy (MRS) provides a non-invasive method for investigating low concentration metabolites, such as γ-aminobutyric acid (GABA). The ISBI Edited MRS Reconstruction Challenge aims at accelerating edited-MRS scans through machine learning models that reconstruct high-quality spectra using four times less data than standard scans. It is composed of three tracks: simulated data, single-vendor, and multi-vendor in-vivo data, each with edited ON and OFF spectra from GABA-edited MEGA-PRESS scans.

This work presents a deep learning method for reconstruction of edited MRS spectra capable of operating with an arbitrary number of available measurement repetitions. It proposes to compute the sample covariance matrix of the measurements and use it as the input of a convolutional neural network (CNN) to extract relevant signal features and produce a high-quality spectrum. The results indicate that the method can perform effectively even with highly noisy data obtained from a single acquisition, and its performance can be further enhanced with multiple acquisitions.


## Overview

This repository consists of following Python scripts:
* The `main.py` implements the pipeline to train and test deep learning approaches for edited MRS.
* The `sweep.py` defines ranges to sweep for optimal hyperparamters using Weights & Biases.
* The `auxiliary/` folder holds helpful functionalities.
  * The `gpu_config.py` defines the GPU configuration.
  * The `utils.py` defines some helpful functions.
* The `basis/` folder holds a edited basis set obtain with FID-A.
* The `challenge/` folder holds some files provided by the ISBI Edited MRS Reconstruction Challenge organizers.
  * The `metric_calculator.py` calculates the metric for the challenge.
  * The `spec_rec.py` defines a spectral registration of the spectra.
  * The `utils.py` defines some helpful functions.
* The `moduls/` folder holds the main scripts for loading, augmenting, and training the data.
  * The `augmentations.py` defines all augmentation steps for the data.
  * The `dataloader.py` loads the datasets and pre-processes the data.
  * The `datamodule.py` defines the data modules, crearting training/validation/testing spilts and allows augmenting the batches during training.
  * The `losses.py` defines the various training losses, such as rangeMAE, SNR, etc.
  * The `models.py` holds the neural architectures.
* The `simulation/` folder holds the scripts to simulate edited MRS spectra.
  * The `basis.py` defines the basis set class to hold the spectra.
  * The `sigModel.py` defines the signal model to simulate the spectra.
  * The `simulation.py` draws simulation parameters from distibutions to allow simulation with the signal model.
  * The `simulationDefs.py` holds the simulation parameters ranges.



## Requirements

| Module            | Version |
|:------------------|:-------:|
| scipy             | 1.10.1  |
| h5py              |  3.8.0  |
| pandas            |  2.0.1  |
| matplotlib        |  3.7.1  |
| numpy             | 1.24.3  |
| pytorch_lightning |  2.0.2  |
| torch             | 1.13.1  |
| wandb             | 0.15.2  |
