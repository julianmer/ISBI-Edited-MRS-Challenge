####################################################################################################
#                                             main.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: S. Amirrajab (s.amirrajab@tue.nl),                                                      #
#          J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 16/02/23                                                                                #
#                                                                                                  #
# Purpose: Train a set of neural models to sweep for optimal hyperparams for the purpose           #
#          of the Edited-MRS Reconstruction Challenge 2023.                                        #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import argparse
import pytorch_lightning as pl
import torch
import wandb

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

# own
from auxiliary.gpu_config import set_gpu_usage
from auxiliary.utils import *
from challenge.metric_calculator import calculate_metrics, calculate_linewidth, calculate_snr
from challenge.spec_reg import basic_spectral_registration
from challenge.utils import *
from moduls.augmentations import *
from moduls.dataloader import *
from moduls.datamodule import *
from moduls.models import *
from moduls.losses import *



#**************************************************************************************************#
#                                             Pipeline                                             #
#**************************************************************************************************#
#                                                                                                  #
# The pipeline allowing to load, augment, train, and test methods.                                 #
#                                                                                                  #
#**************************************************************************************************#
class Pipeline():

    #***************#
    #   main init   #
    #***************#
    def __init__(self):
        self.default_config = {
            'save_path': '',   # path to saved model (for loading)
            'track': 'track1',  # 'track1', 'track2', or 'track3'
            'training_data': 'track1',   # 'track2', 'track3', or 'both' (only for track 2 and 3)
            'sims': 'chal',  # 'chal', or 'own' for challenge or own simulations
            'model': 'covCNN',
            'feedforward': 0,   # set to number of times an output is fed back into the model
            'parallel': False,   # set to True to use net in parallel for transients
            'mean': False,   # if True, use mean of transients
            'transform': 'all',   # 'none', 'baseline', 'custom', 'broadening', ...
            'loss': 'MAE',  # 'rangeMAE', 'rangeMSE', 'MAE', 'MSE', 'total'
            'data_format': 'real',   # 'real', 'complex', 'complexX'
            'transients': 1,
            'epoch': 100,
            'lr': 1e-03,
            'batch_size': 16,
            'activation': 'relu',   # 'tanh', 'relu', 'leaky_relu', 'sigmoid'
            'dropout': 0.0,
            'l1_reg': 0.0,
            'l2_reg': 0.0,
            'num_workers': 4,
            'shuffle': True,
            'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
            'selection': set_gpu_usage() if torch.cuda.is_available() else 0,
            'load_model': False,
            'skip_train': False,
            'skip_test': False,
            'quick_test': True,
            'skip_sub': True,
            'debug_mode': False,
            'log_train': False,
            'log_plot_freq': 400,
        }
        torch.set_num_threads(8)   # limit number of threads if run is CPU heavy


    #**********************************#
    #   switch model based on config   #
    #**********************************#
    def getModel(self, config):
        if config.model == 'baseline':
            model = UNET(config.transients).float()
        elif config.model == 'covCNN' or config.model == 'covCNNClean':
            model = CovCNN(config.dropout, config.activation)
        elif config.model == 'covCNNcomplex' or config.model == 'covCNNcomplexClean':
            model = CovCNNComplex(config.dropout)
        elif config.model == 'covCNNandRNN' or config.model == 'covCNNandRNNClean':
            model = CovCNNandRNN(config.dropout)
        elif config.model == 'covCNNalt' or config.model == 'covCNNaltClean':
            model = CovCNNalt(config.dropout)
        elif config.model == 'covCandRNN' or config.model == 'covCandRNNClean':
            model = CovCandRNN(config.dropout)
        elif config.model == 'covCandRNNcomplex':
            model = CovCandRNNComplex(config.dropout)
        elif 'ResNet' in config.model:
            if 'Rand' in config.model:
                resnet_n_blocks = int(config.model.replace('RandResNet', ''))
                model = ResNet(resnet_n_blocks, config.transients, rand=True)
            else:
                resnet_n_blocks = int(config.model.replace('ResNet', ''))
                model = ResNet(resnet_n_blocks, config.transients, rand=False)
        else:
            raise ValueError('model %s is not recognized' % config.model)
        return model


    #*****************************************#
    #   switch augmentation based on config   #
    #*****************************************#
    def getTransform(self, config):
        # order of applying transformations is important!
        if config.transform == 'baseline':
            transform = [BaselineAugmentaion(config.transients)]
        elif config.transform == 'testing':
            transform = [
                # RandomFrequencyShift(mean=0, std=10, transients=config.transients),
                # RandomPhaseShift(mean=0, std=10 * np.pi / 180, transients=config.transients),
                # RandomNoise(mean=0, std=3, transients=config.transients),

                RandomMMPeak(ppm=[0.93, 1.24, 1.43, 1.72, 2.05, 2.29, 3.0, 3.20, 3.8, 4.3],
                             amp=(15, 15), gamma=(5, 25), sigma=(5, 25), transients=config.transients),

                # RandomLineBroad(mean=1, std=3, transients=config.transients, filter='l2g')

            ]
        elif config.transform == 'custom':
            transform = [
                RandomFrequencyShift(mean=0, std=20, transients=config.transients),
                RandomPhaseShift(mean=0, std=30 * np.pi / 180, transients=config.transients),
                RandomNoise(mean=0, std=10, transients=config.transients),
            ]
        elif config.transform == 'broadening':
            transform = [
                RandomFrequencyShift(mean=0, std=20, transients=config.transients),
                RandomPhaseShift(mean=0, std=30 * np.pi / 180, transients=config.transients),
                RandomNoise(mean=0, std=10, transients=config.transients),
                RandomLineBroad(mean=1, std=10, transients=config.transients, filter='l2g')
            ]
        elif config.transform == 'MMpeaks':
            transform = [
                RandomMMPeak(ppm=[0.93, 1.24, 1.43, 1.72, 2.05, 2.29, 3.0, 3.20, 3.8, 4.3],
                             amp=(0.1, 5), gamma=(5, 25), sigma=(5, 25), transients=config.transients),
                RandomFrequencyShift(mean=0, std=20, transients=config.transients),
                RandomPhaseShift(mean=0, std=30 * np.pi / 180, transients=config.transients),
                RandomNoise(mean=0, std=10, transients=config.transients),
            ]
        elif config.transform == 'all':
            transform = [
                RandomMMPeak(ppm=[0.93, 1.24, 1.43, 1.72, 2.05, 2.29, 3.0, 3.20, 3.8, 4.3],
                             amp=(0.1, 5), gamma=(5, 25), sigma=(5, 25), transients=config.transients),
                RandomFrequencyShift(mean=0, std=20, transients=config.transients),
                RandomPhaseShift(mean=0, std=30 * np.pi / 180, transients=config.transients),
                RandomNoise(mean=0, std=10, transients=config.transients),
                RandomLineBroad(mean=1, std=10, transients=config.transients, filter='l2g')
            ]
        elif config.transform == 'none':
            transform = None
        else:
            raise ValueError('transform %s is not recognized' % config.transform)
        return transform


    #*********************************#
    #   switch loss based on config   #
    #*********************************#
    def getLoss(self, config):
        if config.loss.lower() == 'mse':
            loss = MSELoss()
        elif config.loss.lower() == 'mae':
            loss = MAELoss()
        elif config.loss.lower() == 'rangemse':
            loss = RangeMSELoss(noise=True)      # noise=False means the range does not
        elif config.loss.lower() == 'rangemae':   # include the noise region (9.8-10.8 ppm)
            loss = RangeMAELoss(noise=True)
        elif config.loss.lower() == 'total':
            loss = TotalLoss()
        elif config.loss.lower() == 'expmse':
            loss = ExpLoss()
        elif config.loss.lower() == 'custom':
            loss = CustomLoss()
        else:
            raise ValueError('loss %s is not recognized' % config.loss)
        return loss


    #********************************#
    #   main pipeline for training   #
    #********************************#
    def main(self, config=None):
        # wandb init
        if (config is None) or config['online']:
            wandb.init(config=config)
            wandb_logger = WandbLogger()
        else:
            wandb.init(mode='disabled', config=config)
            wandb_logger = None
        config = wandb.config
        define_metrics()  # init metrics for wandb

        # combine default configs and wandb config
        parser = argparse.ArgumentParser()
        for keys in self.default_config:
            parser.add_argument('--' + keys, default=self.default_config[keys],
                                type=type(self.default_config[keys]))
        args = parser.parse_args()
        config.update(args)

        # switch data module based on track
        if config.track == 'track1':
            data = DataModuleTrack1(config, self.getTransform(config))
        elif config.track == 'track2':
            data = DataModuleTrack2(config, self.getTransform(config), trainingData=config.training_data)
        elif config.track == 'track3':
            data = DataModuleTrack3(config, self.getTransform(config), trainingData=config.training_data)
        else:
            raise ValueError('unrecognized track: %s' % config.transform)

        # model inits
        self.net = self.getModel(config)
        self.loss_fn = self.getLoss(config)
        self.optim = torch.optim.Adam

        self.model = Framework(self.net, self.loss_fn, config, self.optim, config.lr)

        # callbacks, etc.
        checkpoint_callback = ModelCheckpoint(monitor='val_loss')
        early_stop_callback = EarlyStopping(monitor="val_loss", mode="min",
                                            min_delta=0.0, patience=10)
        # gpu acceleration
        if config.device == 'cuda':
            # matrix multiplications use the bfloat16
            try: torch.set_float32_matmul_precision('medium')
            except: print('bfloat16 for matmul not supported')
            accelerator = 'gpu'
            devices = [config.selection]   # select gpu by idx or multiple gpus by range or idxs
        else: accelerator, devices = None, None

        # load model...
        if config.load_model:
            self.model.load_from_checkpoint(config.save_path, net=self.net, loss=self.loss_fn,
                                            config=config)
        # ...train model
        if not config.skip_train:
            trainer = pl.Trainer(max_epochs=config.epoch,
                                 accelerator=accelerator,
                                 devices=devices,
                                 logger=wandb_logger,
                                 callbacks=[checkpoint_callback],  # , early_stop_callback],
                                 # precision=16,
                                 # detect_anomaly=True,
                                 # gradient_clip_val=0.1
                                 )
            # wrapper = Framework(self.model.net, self.loss_fn, config, self.optim, config.lr,
            #                     specRecNN=True)
            # self.model = wrapper

            trainer.fit(self.model, data)

            # load best model
            self.model.load_from_checkpoint(trainer.checkpoint_callback.best_model_path,
                                            net=self.net, loss=self.loss_fn, config=config,
                                            # specRecNN=True,
                                            )
        # ...or only test model
        else: data.setup('test')

        if not config.skip_test:
            # quick test
            if config.quick_test:
                self.model.test_loop(data.test_dataloader())

            # test models
            else:
                for id, set in data.test_sets().items():
                    self.model.identifier = id
                    self.model.test_loop(set)

                for id, set in data.test_sets_noGT().items():
                    self.model.identifier = id
                    self.model.test_loop_noGT(set)

        # create submission
        if not config.skip_sub:
            if config.track == 'track3':
                pred2048, ppm2048 = self.model.submission(data.submission_set(type='2048'))
                pred4096, ppm4096 = self.model.submission(data.submission_set(type='4096'))

                # upsample the 4096 data
                time = False   # upsample in time domain
                if time: pred4096 = torch.fft.irfft(pred4096, n=2048, dim=1)

                pred4096 = torch.nn.functional.interpolate(pred4096.unsqueeze(1), size=4096,
                                                           mode='linear').squeeze()
                ppm4096 = torch.nn.functional.interpolate(ppm4096.unsqueeze(1), size=4096,
                                                          mode='linear').squeeze()
                # to freq domain
                if time: pred4096 = torch.fft.rfft(pred4096, n=4096, dim=1)

                save_submission_track3(pred2048.detach().numpy(), ppm2048.detach().numpy(),
                                       pred4096.detach().numpy(), ppm4096.detach().numpy(),
                                       './submission/' + config.track + '.h5')
            else:
                pred, ppm = self.model.submission(data.submission_set())
                save_submission(pred.detach().numpy(), ppm.detach().numpy(),
                                './submission/' + config.track + '.h5')
            print('Submission saved to ./submission/' + config.track + '.h5')

        wandb.finish()


#**************************************************************************************************#
#                                          Class Framework                                         #
#**************************************************************************************************#
#                                                                                                  #
# The framework allowing to define, train, and test data-driven models.                            #
#                                                                                                  #
#**************************************************************************************************#
class Framework(pl.LightningModule):
    def __init__(self, net, loss, config, opt=torch.optim.Adam, lr=1e-3, specRecNN=False):
        super().__init__()
        self.net = net
        self.loss = loss
        self.config = config
        self.opt = opt
        self.lr = lr

        self.specRecMB = False
        self.specRecNN = specRecNN

        if self.specRecNN:
            self.reg = Custom()

        self.identifier = ''

    def forward(self, x, ppm=None, t=None):
        if self.config.mean:
            x = x.mean(dim=-1, keepdim=True)

        if self.config.parallel:   # parallelize the model
            outs = [self.net(x[..., i].unsqueeze(-1)).unsqueeze(-1) for i in range(x.shape[-1])]
            out = torch.stack(outs, dim=-1).squeeze()
        else:   # standard forward pass
            out = self.net(x)

        # feed through multiple times
        for _ in range(self.config.feedforward):
            if self.config.parallel:
                out = out[..., torch.randperm(out.shape[-1])]  # random permutation of the transients
                outs = [self.net(out[..., i].unsqueeze(-1)).unsqueeze(-1)
                        for i in range(out.shape[-1])]
                out = torch.stack(outs, dim=-1).squeeze()
            else:
                out = self.net(out.unsqueeze(-1))

        if self.config.parallel:
            # spectral registration
            if self. specRecMB:
                out = basic_spectral_registration(out.unsqueeze(2).cpu().detach().numpy(),
                                                  t.cpu().detach().numpy())

                out = torch.Tensor(out).to(self.device).squeeze()

            if self.specRecNN:
                # freeze network
                for param in self.net.parameters():
                    param.requires_grad = False

                # spectral registration with NN
                out = self.reg(out, t)

            # average over the transients
            out = torch.mean(out, dim=-1).squeeze(-1)

        # if type(ppm) is torch.Tensor:  # mask spectra outside of range of ppm [2.5, 4] with zero
        #     mask = torch.full(out.shape, 1./torch.finfo(torch.float16).max, device=out.device)
        #     min_ind, max_ind = torch.argmin(ppm[ppm >= 4]), torch.argmin(ppm[ppm >= 2.5])
        #     mask[:, min_ind:max_ind] = out[:, min_ind:max_ind]
        #     out = mask

        return out

    def training_step(self, batch, batch_idx):
        x, y, ppm, t = batch
        y_hat = self(x, ppm, t)
        loss = self.loss(y_hat, y, ppm)
        self.log("train_loss", loss.item())

        # log plot only once every n-th epoch based in log_plot_freq
        if self.config.log_train and  batch_idx == 0 and \
                (self.current_epoch + 1) % self.config.log_plot_freq == 0:
            fig = plot_spectra_batch(batch, self, show=False)
            wandb.log({f"GT vs. Recon, train epoch {self.current_epoch + 1}": fig})

        # add l1 or l2 regularization
        l1_norm = sum(p.abs().sum() for p in self.net.parameters())
        l2_norm = sum(p.pow(2.0).sum() for p in self.net.parameters())
        loss += self.config.l1_reg * l1_norm + self.config.l2_reg * l2_norm
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, ppm, t = batch
        y_hat = self(x, ppm, t)
        loss = self.loss(y_hat, y, ppm)
        self.log("val_loss", loss.item(), prog_bar=True)

        # log plot only once every n-th epoch based in log_plot_freq
        if batch_idx == 0 and (self.current_epoch + 1) % self.config.log_plot_freq == 0:
            fig = plot_spectra_batch(batch, self, show=False)
            wandb.log({f"GT vs. Recon, val epoch {self.current_epoch + 1}": fig})

    def test_step(self, batch, batch_idx):
        x, y, ppm, t = batch
        y_hat = self(x, ppm, t)
        loss = self.loss(y_hat, y, ppm)
        wandb.log({self.identifier + "test_loss": loss.item()})

        # compute quality metrics
        np_pred = y_hat.float().cpu().detach().numpy()
        np_y_test = y.float().cpu().detach().numpy()
        np_ppm = ppm.float().cpu().detach().numpy()
        model_metrics = calculate_metrics(np_pred, np_y_test, np_ppm)

        # visulize predictions, ground truth, and residuals in same plot with residual pushed up
        if True:
            max_ind = np.amax(np.where(np_ppm[0] >= -5))
            min_ind = np.amin(np.where(np_ppm[0] <= 10))

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
            ax.plot(np_ppm[0, min_ind:max_ind], np_pred[0, min_ind:max_ind], label='pred')
            ax.plot(np_ppm[0, min_ind:max_ind], np_y_test[0, min_ind:max_ind], label='gt')
            ax.plot(np_ppm[0, min_ind:max_ind], np_pred[0, min_ind:max_ind] - np_y_test[0, min_ind:max_ind] + 0.3, 'r', label='residual')
            ax.legend()

            # have axis go from 4 to 2.5
            ax.invert_xaxis()

            plt.show()
            stop

        for key, value in model_metrics.items():
            print(f"{self.identifier + key}: {value}")
            wandb.log({self.identifier + key: value})

        # log plot only for first batch in test set
        if batch_idx == 0:
            fig = plot_spectra_batch(batch, self, show=False)
            wandb.log({f"GT vs. Recon, {self.identifier}test batch {batch_idx + 1}": fig})

            figX, figY = plot_covs_batch(batch, show=False)
            wandb.log({f"Covariance Kx, {self.identifier}test batch {batch_idx + 1}": figX})
            wandb.log({f"Covariance Ky, {self.identifier}test batch {batch_idx + 1}": figY})

    def test_loop(self, dataloader):
        for batch_idx, batch in enumerate(dataloader):
            self.test_step(batch, batch_idx)

    def test_step_noGT(self, batch, batch_idx):
        x, ppm, t = batch
        y_hat = self(x, ppm, t)

        # compute quality metrics
        np_pred = y_hat.float().cpu().detach().numpy()
        np_ppm = ppm.float().cpu().detach().numpy()
        wandb.log({self.identifier + 'snr': calculate_snr(np_pred, np_ppm)})
        wandb.log({self.identifier + 'linewidth': calculate_linewidth(np_pred, np_ppm)})

        # log plot only for first batch in test set
        if batch_idx == 0:
            fig = plot_spectrum(np_pred, np_ppm, show=False)
            wandb.log({f"Recon, {self.identifier}test batch {batch_idx + 1}": fig})

    def test_loop_noGT(self, dataloader):
        for batch_idx, batch in enumerate(dataloader):
            self.test_step_noGT(batch, batch_idx)

    def submission(self, dataloader):
        batch_idx, batch = next(enumerate(dataloader))   # make sure all samples are in one batch
        x, ppm, t = batch
        return self(x, ppm, t), ppm

    def configure_optimizers(self):
        params = [param for param in self.parameters() if param.requires_grad]
        optimizer = self.opt(params, lr=self.lr)
        return optimizer

    # in order to improve performance, override optimizer_zero_grad()
    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)


#**********#
#   main   #
#**********#
if __name__ == "__main__":

    config = {"online": False}

    Pipeline().main({"online": False})

