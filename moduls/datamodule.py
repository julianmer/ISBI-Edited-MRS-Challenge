####################################################################################################
#                                         datamodule.py                                            #
####################################################################################################
#                                                                                                  #
# Authors: J. P. Merkofer (j.p.merkofer@tue.nl),                                                   #
#                                                                                                  #
# Created: 03/03/23                                                                                #
#                                                                                                  #
# Purpose: Definition of data modules, taking care of loading and processing the data.             #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import pytorch_lightning as pl
import torch

from torch.utils.data import DataLoader

# own
from moduls.augmentations import *
from moduls.dataloader import *


#**************************************************************************************************#
#                                      Class DataModuleTrack1                                      #
#**************************************************************************************************#
#                                                                                                  #
# The data module for training, validation, testing data.                                          #
#                                                                                                  #
#**************************************************************************************************#
class DataModuleTrack1(pl.LightningDataModule):
    def __init__(self, config, transform=None):
        super().__init__()
        self.config = config
        self.transform = transform

    def setup(self, stage=None):

        if 'clean' in self.config.model.lower(): tf = None   # model is trained with clean data
        else: tf = self.transform

        if self.config.sims == 'chal':
            self.train_dataset = SynthDataset(
                transform=tf,
                data_format=self.config.data_format,
                phase='train',
                transients=self.config.transients,
            )
        else:
            self.train_dataset = SimulatedDataset(
                transform=tf,
                data_format=self.config.data_format,
                transients=self.config.transients,
            )

        self.valid_dataset = SynthDataset(
            transform=self.transform,
            data_format=self.config.data_format,
            phase='val',
            transients=self.config.transients,
        )
        self.test_dataset = SynthDataset(
        transform=self.transform,
        data_format=self.config.data_format,
        phase='test',
        transients=self.config.transients,
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.config.batch_size,
                          shuffle=self.config.shuffle,
                          num_workers=self.config.num_workers,
                          pin_memory=torch.cuda.is_available(),
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, self.config.batch_size,
                          num_workers=self.config.num_workers,
                          pin_memory=torch.cuda.is_available(),
                          persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.config.batch_size,
                          num_workers=self.config.num_workers,
                          pin_memory=torch.cuda.is_available(),
                          persistent_workers=True)

    def test_sets(self):
        # test model with test slice of same data
        test = self.test_dataloader()

        # test model with simulated baseline test data
        self.test_dataset.transform = [BaselineAugmentaion(self.config.transients)]
        cha_sim = self.test_dataloader()

        # test model with in-vivo data (from track 2)
        test_dataset = RealDataset(
            data_format=self.config.data_format,
            phase='none',  # all
            track='both'  # track 2, 3, or both
        )
        in_vivo = DataLoader(test_dataset, self.config.batch_size,
                             num_workers=self.config.num_workers)

        return {'': test, 'cha-sims ': cha_sim, 'in-vivo ': in_vivo}

    def test_sets_noGT(self):
        # test model with simulated submission data (no gt)
        test_dataset = SubmissionDataset(
            '../../../data/Edited-MRS Reconstruction Challenge Data/track_01_test_data.h5',
            data_format=self.config.data_format,
            phase='none',
            transients=self.config.transients,
        )
        track1 = DataLoader(test_dataset, self.config.batch_size,
                            num_workers=self.config.num_workers)

        # test model with in-vivo submission data (no gt)
        test_dataset = SubmissionDataset(
            '../../../data/Edited-MRS Reconstruction Challenge Data/track_02_test_data.h5',
            data_format=self.config.data_format,
            phase='none',
            transients=self.config.transients,
        )
        track2 = DataLoader(test_dataset, self.config.batch_size,
                            num_workers=self.config.num_workers)

        return {'sub-track1 ': track1, 'sub-track2 ': track2}

    def submission_set(self):
        sub_dataset = SubmissionDataset(
            '../../../data/Edited-MRS Reconstruction Challenge Data/track_01_test_data.h5',
            data_format=self.config.data_format,
            phase='none',
            transients=self.config.transients,
        )
        sub_dataset = DataLoader(sub_dataset, len(sub_dataset),
                                  num_workers=self.config.num_workers)
        return sub_dataset


#**************************************************************************************************#
#                                      Class DataModuleTrack2                                      #
#**************************************************************************************************#
#                                                                                                  #
# The data module for training, validation, testing data for track 2.                              #
#                                                                                                  #
#**************************************************************************************************#
class DataModuleTrack2(pl.LightningDataModule):
    def __init__(self, config, transform=None, trainingData='track2'):
        super().__init__()
        self.config = config
        self.transform = transform
        self.trainingData = trainingData    # track2, track3, or both

    def setup(self, stage=None):

        if 'clean' in self.config.model.lower():
            tf = None   # model is trained with clean data
            clean = True
        else:
            tf = self.transform
            clean = False

        self.train_dataset = RealDataset(
            transform=tf,
            data_format=self.config.data_format,
            phase='train',
            transients=self.config.transients,
            track=self.trainingData,
            clean=clean   # model is trained with clean data
        )
        self.valid_dataset = RealDataset(
            transform=self.transform,
            data_format=self.config.data_format,
            phase='val',
            transients=self.config.transients,
            track=self.trainingData
        )
        self.test_dataset = RealDataset(
            transform=self.transform,
            data_format=self.config.data_format,
            phase='test',
            transients=self.config.transients,
            track=self.trainingData
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, self.config.batch_size,
                          shuffle=self.config.shuffle,
                          num_workers=self.config.num_workers,
                          pin_memory=torch.cuda.is_available(),
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, self.config.batch_size,
                          num_workers=self.config.num_workers,
                          pin_memory=torch.cuda.is_available(),
                          persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, self.config.batch_size,
                          num_workers=self.config.num_workers,
                          pin_memory=torch.cuda.is_available(),
                          persistent_workers=True)

    def test_sets(self):
        # test model with test slice of same data
        test = self.test_dataloader()

        # # test model with simulated baseline test data
        # test_dataset = SynthDataset(
        #     transform=[BaselineAugmentaion(self.config.transients)],
        #     data_format=self.config.data_format,
        #     phase='test',
        #     transients=self.config.transients,
        # )
        # cha_sim = DataLoader(test_dataset, self.config.batch_size,
        #                      num_workers=self.config.num_workers)

        return {'': test}#, 'cha-sims ': cha_sim}

    def test_sets_noGT(self):
        # test model with simulated submission data (no gt)
        test_dataset = SubmissionDataset(
            '../../../data/Edited-MRS Reconstruction Challenge Data/track_01_test_data.h5',
            data_format=self.config.data_format,
            phase='none',
            transients=self.config.transients,
        )
        track1 = DataLoader(test_dataset, self.config.batch_size,
                            num_workers=self.config.num_workers)

        # test model with in-vivo submission data (no gt)
        test_dataset = SubmissionDataset(
            '../../../data/Edited-MRS Reconstruction Challenge Data/track_02_test_data.h5',
            data_format=self.config.data_format,
            phase='none',
            transients=self.config.transients,
        )
        track2 = DataLoader(test_dataset, self.config.batch_size,
                            num_workers = self.config.num_workers)

        return {'sub-track1 ': track1, 'sub-track2 ': track2}

    def submission_set(self, type=None):
        sub_dataset = SubmissionDataset(
            '../../../data/Edited-MRS Reconstruction Challenge Data/track_02_test_data.h5',
            data_format=self.config.data_format,
            phase='none',
            transients=self.config.transients,
            type=type
        )
        sub_dataset = DataLoader(sub_dataset, len(sub_dataset),
                                  num_workers=self.config.num_workers)
        return sub_dataset


#**************************************************************************************************#
#                                      Class DataModuleTrack3                                      #
#**************************************************************************************************#
#                                                                                                  #
# The data module for training, validation, testing data for track 3.                              #
#                                                                                                  #
#**************************************************************************************************#
class DataModuleTrack3(DataModuleTrack2):
    def __init__(self, config, transform=None, trainingData='track3'):
        DataModuleTrack2.__init__(self, config, transform, trainingData)

    def test_sets_noGT(self):
        # test model with simulated submission data (no gt)
        test_dataset = SubmissionDataset(
            '../../../data/Edited-MRS Reconstruction Challenge Data/track_01_test_data.h5',
            data_format=self.config.data_format,
            phase='none',
            transients=self.config.transients,
        )
        track1 = DataLoader(test_dataset, self.config.batch_size,
                            num_workers=self.config.num_workers)

        # test model with in-vivo submission data (no gt)
        test_dataset = SubmissionDataset(
            '../../../data/Edited-MRS Reconstruction Challenge Data/track_03_test_data.h5',
            data_format=self.config.data_format,
            phase='none',
            transients=self.config.transients,
            type='2048'
        )
        track3_2048 = DataLoader(test_dataset, self.config.batch_size,
                                 num_workers = self.config.num_workers)

        # test model with in-vivo submission data (no gt)
        test_dataset = SubmissionDataset(
            '../../../data/Edited-MRS Reconstruction Challenge Data/track_03_test_data.h5',
            data_format=self.config.data_format,
            phase='none',
            transients=self.config.transients,
            type='4096'
        )
        track3_4096 = DataLoader(test_dataset, self.config.batch_size,
                                 num_workers=self.config.num_workers)

        return {'sub-track1 ': track1, 'sub-track3_2048 ': track3_2048,
                'sub-track3_4096 ': track3_4096}

    def submission_set(self, type=None):
        sub_dataset = SubmissionDataset(
            '../../../data/Edited-MRS Reconstruction Challenge Data/track_03_test_data.h5',
            data_format=self.config.data_format,
            phase='none',
            transients=self.config.transients,
            type=type
        )
        sub_dataset = DataLoader(sub_dataset, len(sub_dataset),
                                  num_workers=self.config.num_workers)
        return sub_dataset