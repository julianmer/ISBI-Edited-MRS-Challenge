####################################################################################################
#                                            sweep.py                                              #
####################################################################################################
#                                                                                                  #
# Authors: S. Amirrajab (s.amirrajab@tue.nl)                                                       #
#                                                                                                  #
# Created: 23/01/23                                                                                #
#                                                                                                  #
# Purpose: Sweep parameter definition for Weights & Biases.                                        #
#                                                                                                  #
####################################################################################################

if __name__ == "__main__":

    #*************#
    #   imports   #
    #*************#
    import pytorch_lightning as pl
    import wandb

    # own
    from main import *


    #**************************#
    #   eliminate randomness   #
    #**************************#
    pl.seed_everything(42)


    #************************************#
    #   configure sweep and parameters   #
    #************************************#
    sweep_config = {"method": "grid"}

    metric = {
        "name": "val_loss",
        "goal": "minimize",

        "additional_metrics": {
            "name": "test_loss",
            "goal": "minimize"
        }
    }

    sweep_parameters = {
        "epoch":        {"values": [10000]},
        "lr":            {"values": [1e-3]},
        "batch_size":    {"values": [16]},
        "activation":    {"values": ['relu']},
        "dropout": {"values": [0.0]},
        "l1_reg": {"values": [0.0]},
        "l2_reg": {"values": [0.0]},
        "model": {"values": ['covCandRNN']},
        "data_format": {"values": ['real']},
        "mean": {"values": [False]},
        "transform": {"values": ['none']},
        "loss": {"values": ['MAE']},
    }

    sweep_config["name"] = "model_sweep"   # sweep name
    sweep_config["parameters"] = sweep_parameters   # add parameters to sweep
    sweep_config["metric"]= metric    # add metric to sweep


    # create sweep ID and name project
    wandb.login(key='')   # TODO: add your key here
    sweep_id = wandb.sweep(sweep_config, project="", entity="")   # TODO: add your project
                                                                  # and entity here

    # training the model
    pipeline = Pipeline()
    wandb.agent(sweep_id, pipeline.main)
