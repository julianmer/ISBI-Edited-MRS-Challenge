####################################################################################################
#                                            models.py                                             #
####################################################################################################
#                                                                                                  #
# Authors: S. Amirrajab (s.amirrajab@tue.nl),                                                      #
#          J. P. Merkofer (j.p.merkofer@tue.nl)                                                    #
#                                                                                                  #
# Created: 15/08/22                                                                                #
#                                                                                                  #
# Purpose: Definitions of various neural networks to be used in the Edited-MRS Reconstruction      #
#          Challenge 2023.                                                                         #
#                                                                                                  #
####################################################################################################


#*************#
#   imports   #
#*************#
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


#**************************************************************************************************#
#                                             Class UNET                                           #
#**************************************************************************************************#
#                                                                                                  #
# 2D U-net to combine the transients, considering the transients as a 2D "image"                   #
# (spectral_points x transients).                                                                  #
#                                                                                                  #
#**************************************************************************************************#
class UNET(nn.Module):

    # initializing the weights for the convolution layers
    def __init__(self,transient_count):
        super(UNET,self).__init__()

        self.down_conv_1_1 = nn.Conv2d(1,16,kernel_size=(5,1),padding="same")
        self.down_conv_1_2 = nn.Conv2d(16,16,kernel_size=(3,3),padding="same")

        self.down_conv_2_1 = nn.Conv2d(16,32,kernel_size=(3,3),padding="same")
        self.down_conv_2_2 = nn.Conv2d(32,32,kernel_size=(3,3),padding="same")

        self.down_conv_3_1 = nn.Conv2d(32,64,kernel_size=(3,3),padding="same")
        self.down_conv_3_2 = nn.Conv2d(64,64,kernel_size=(3,3),padding="same")

        self.up_conv_1_1 = nn.Conv2d(64,128,kernel_size=(3,3),padding="same")
        self.up_conv_1_2 = nn.Conv2d(128,128,kernel_size=(3,3),padding="same")

        self.up_conv_2_1 = nn.Conv2d(192,64,kernel_size=(3,3),padding="same")
        self.up_conv_2_2 = nn.Conv2d(64,64,kernel_size=(3,3),padding="same")

        self.up_conv_3_1 = nn.Conv2d(96,32,kernel_size=(3,3),padding="same")
        self.up_conv_3_2 = nn.Conv2d(32,32,kernel_size=(3,3),padding="same")

        self.end_conv_1_1 = nn.Conv2d(48,128,kernel_size=(1,transient_count))
        self.end_conv_1_2 = nn.Conv2d(128,1,kernel_size=(5,5),padding="same")
    
    # defining forward pass
    def forward(self,x):
        x = torch.unsqueeze(x, 1).float()

        # changing order of dimensions, as in torch the filters come first
        # y = x.transpose(1,3)
        # y = y.transpose(2,3)

        y = F.relu(self.down_conv_1_1(x))
        y_skip1 = F.relu(self.down_conv_1_2(y))

        y = F.max_pool2d(y_skip1,(2,1))

        y = F.relu(self.down_conv_2_1(y))
        y_skip2 = F.relu(self.down_conv_2_2(y))

        y = F.max_pool2d(y_skip2,(2,1))

        y = F.relu(self.down_conv_3_1(y))
        y_skip3 = F.relu(self.down_conv_3_2(y))

        y = F.max_pool2d(y_skip3,(2,1))

        y = F.relu(self.up_conv_1_1(y))
        y = F.relu(self.up_conv_1_2(y))

        y = F.interpolate(y,scale_factor=(2,1))

        y = torch.concat([y,y_skip3],axis=1)

        y = F.relu(self.up_conv_2_1(y))
        y = F.relu(self.up_conv_2_2(y))

        y = F.interpolate(y,scale_factor=(2,1))

        y = torch.concat([y,y_skip2],axis=1)

        y = F.relu(self.up_conv_3_1(y))
        y = F.relu(self.up_conv_3_2(y))

        y = F.interpolate(y,scale_factor=(2,1))

        y = torch.concat([y,y_skip1],axis=1)

        y = F.relu(self.end_conv_1_1(y))
        y = self.end_conv_1_2(y)

        # converting the order of layers back to the original format

        y = y.transpose(1,3)
        y = y.transpose(1,2)

        # flattening result to only have 2 dimensions
        return y.view(y.shape[0],-1)


#**************************************************************************************************#
#                                        Class ResidualBlock                                       #
#**************************************************************************************************#
#                                                                                                  #
# A residual block for the use in a ResNet.                                                        #
#                                                                                                  #
#**************************************************************************************************#
class ResidualBlock(nn.Module):
    def __init__(self, in_features, norm='batch', activ='relu', kernel_size=3):
        super(ResidualBlock, self).__init__()

        if norm== 'batch':
            normalization = nn.BatchNorm1d(in_features)
        elif norm == 'instance':
            normalization = nn.InstanceNorm1d(in_features)
        else:
            raise ValueError('normalization layer %s is not recognized' % norm)

        if activ=='relu':
            activation = nn.ReLU(inplace=True)
        elif activ == 'leakyrelu':
            activation = nn.LeakyReLU(inplace=True)

        pw = (kernel_size - 1) // 2
        conv_block = [ 
                        nn.ReflectionPad1d(pw),
                        nn.Conv1d(in_features, in_features, kernel_size),
                        normalization,
                        activation,
                        nn.ReflectionPad1d(pw),
                        nn.Conv1d(in_features, in_features, kernel_size),
                        normalization,
                        activation]
        self.conv_block = nn.Sequential(*conv_block)
    def forward(self, x):
        return x + self.conv_block(x)


#**************************************************************************************************#
#                                            Class ResNet                                          #
#**************************************************************************************************#
#                                                                                                  #
# A ResNNet architecture.                                                                          #
#                                                                                                  #
#**************************************************************************************************#
class ResNet(nn.Module):
    def __init__(self, resnet_n_blocks, in_features, rand=False, kernel_size=3):
        # Initialize the superclass
        super(ResNet, self).__init__()

        self.resnet_n_blocks = resnet_n_blocks
        self.in_features = in_features
        self.rand = rand


        model = []
        for i in range(self.resnet_n_blocks):
            model += [ResidualBlock(in_features, kernel_size=kernel_size)]

        # final output conv
        
        pw = (kernel_size - 1) // 2 
        model += [ nn.ReflectionPad1d(pw),
                  nn.Conv1d(in_features, 1, kernel_size=kernel_size),
                  nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        input = torch.permute(input, (0, 2, 1))
        if self.rand: input = input[:, torch.randperm(input.size()[1])]
        return self.model(input.float()).squeeze()


#**************************************************************************************************#
#                                             Class CNN                                            #
#**************************************************************************************************#
#                                                                                                  #
# A simple convolutional neural network (CNN), followed by dense layers.                           #
# Submission for track 1 of the challenge.                                                         #
#                                                                                                  #
#**************************************************************************************************#
class CovCNN(torch.nn.Module):
    def __init__(self, dropout=0, activation='relu'):
        super(CovCNN, self).__init__()
        self.dropout = dropout

        self.bn1d = torch.nn.BatchNorm2d(num_features=1)
        self.fl = torch.nn.Flatten()

        self.con1 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(9, 9),
                                    stride=(4, 4), )#padding=(2, 2), dilation=(2, 2))
        self.con2 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(7, 7),
                                    stride=(3, 3), )#padding=(2, 2), dilation=(2, 2))
        self.con3 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 5),
                                    stride=(2, 2), )#padding=(2, 2), dilation=(2, 2))
        self.con4 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3),
                                    stride=(1, 1), )#padding=(2, 2), dilation=(2, 2))
        self.fc1 = torch.nn.Linear(in_features=6400, out_features=2048)
        self.fc2 = torch.nn.Linear(in_features=2048, out_features=2048)
        self.fc3 = torch.nn.Linear(in_features=2048, out_features=2048)
        self.out = torch.nn.Linear(in_features=2048, out_features=2048)

        if activation == 'relu': self.act = F.relu
        elif activation == 'sigmoid': self.act = F.sigmoid
        elif activation == 'tanh': self.act = F.tanh
        elif activation == 'leaky_relu': self.act = F.leaky_relu
        elif activation == 'elu': self.act = F.elu
        elif activation == 'selu': self.act = F.selu
        elif activation == 'celu': self.act = F.celu
        elif activation == 'gelu': self.act = F.gelu
        elif activation == 'softplus': self.act = F.softplus
        elif activation == 'softshrink': self.act = F.softshrink
        elif activation == 'softsign': self.act = F.softsign
        elif activation == 'tanhshrink': self.act = F.tanhshrink
        elif activation == 'hardtanh': self.act = F.hardtanh


    def forward(self, x):
        if len(x.shape) > 3: x = x.squeeze()
        x = x.float()
        x = torch.matmul(x, torch.permute(x, (0, 2, 1)))
        x = x.unsqueeze(1)

        x = self.bn1d(x)

        x = self.con1(x)
        x = torch.nn.Dropout(self.dropout)(x)
        x = self.act(x)
        x = self.con2(x)
        x = torch.nn.Dropout(self.dropout)(x)
        x = self.act(x)
        x = self.con3(x)
        x = torch.nn.Dropout(self.dropout)(x)
        x = self.act(x)
        x = self.con4(x)
        x = torch.nn.Dropout(self.dropout)(x)
        x = self.act(x)

        x = self.fl(x)
        x = self.fc1(x)

        x = torch.nn.Dropout(self.dropout)(x)
        x = self.act(x)
        x = self.fc2(x)
        x = torch.nn.Dropout(self.dropout)(x)
        x = self.act(x)
        x = self.fc3(x)
        x = torch.nn.Dropout(self.dropout)(x)
        x = self.act(x)
        x = self.out(x)
        return x
        
        
#**************************************************************************************************#
#                                             Class CNN                                            #
#**************************************************************************************************#
#                                                                                                  #
# A simple convolutional neural network (CNN), followed by dense layers.                           #
#                                                                                                  #
#**************************************************************************************************#
class CovCNNComplex(torch.nn.Module):
    def __init__(self, dropout=0):
        super(CovCNNComplex, self).__init__()
        self.dropout = dropout

        self.bn1d = torch.nn.BatchNorm2d(num_features=2)
        self.fl = torch.nn.Flatten()

        self.con1 = torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(9, 9),
                                    stride=(4, 4), )#padding=(2, 2), dilation=(2, 2))
        self.con2 = torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(7, 7),
                                    stride=(3, 3), )#padding=(2, 2), dilation=(2, 2))
        self.con3 = torch.nn.Conv2d(in_channels=2, out_channels=2, kernel_size=(5, 5),
                                    stride=(2, 2), )#padding=(2, 2), dilation=(2, 2))
        self.con4 = torch.nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(3, 3),
                                    stride=(1, 1), )#padding=(2, 2), dilation=(2, 2))
        self.fc1 = torch.nn.Linear(in_features=6400, out_features=2048)
        self.fc2 = torch.nn.Linear(in_features=2048, out_features=2048)
        self.fc3 = torch.nn.Linear(in_features=2048, out_features=2048)
        self.out = torch.nn.Linear(in_features=2048, out_features=2048)

    def forward(self, x):
        if len(x.shape) > 3: x = x.squeeze()

        x = torch.matmul(x, torch.permute(x, (0, 2, 1)))
        x = x.unsqueeze(1)

        # from complex to real and imag stacked
        x = torch.cat((torch.real(x), torch.imag(x)), dim=1)
        x = x.float()
        
        x = self.bn1d(x)
        x = self.con1(x)
        x = torch.nn.Dropout(self.dropout)(x)
        x = F.relu(x)
        x = self.con2(x)
        x = torch.nn.Dropout(self.dropout)(x)
        x = F.relu(x)
        x = self.con3(x)
        x = torch.nn.Dropout(self.dropout)(x)
        x = F.relu(x)
        x = self.con4(x)
        x = torch.nn.Dropout(self.dropout)(x)
        x = F.relu(x)
        x = self.fl(x)
        x = self.fc1(x)
        x = torch.nn.Dropout(self.dropout)(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.nn.Dropout(self.dropout)(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = torch.nn.Dropout(self.dropout)(x)
        x = F.relu(x)
        x = self.out(x)
        return x


#**************************************************************************************************#
#                                             Class CNN                                            #
#**************************************************************************************************#
#                                                                                                  #
# A simple convolutional neural network (CNN), followed by dense layers.                           #
#                                                                                                  #
#**************************************************************************************************#
class CovCNNandRNN(torch.nn.Module):
    def __init__(self, dropout=0):
        super(CovCNNandRNN, self).__init__()
        self.dropout = dropout

        self.bn2d = torch.nn.BatchNorm2d(num_features=1)
        self.bn1d = torch.nn.BatchNorm1d(num_features=2)
        self.fl = torch.nn.Flatten()

        self.con1 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(9, 9),
                                    stride=(4, 4), )#padding=(2, 2), dilation=(2, 2))
        self.con2 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(7, 7),
                                    stride=(3, 3), )#padding=(2, 2), dilation=(2, 2))
        self.con3 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(5, 5),
                                    stride=(2, 2), )#padding=(2, 2), dilation=(2, 2))
        self.con4 = torch.nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 3),
                                    stride=(1, 1), )#padding=(2, 2), dilation=(2, 2))
        self.fc = torch.nn.Linear(in_features=6400, out_features=2048)
        # self.fc2 = torch.nn.Linear(in_features=2048, out_features=2048)
        # self.fc3 = torch.nn.Linear(in_features=2048, out_features=2048)
        # self.out = torch.nn.Linear(in_features=2048, out_features=2048)

        self.gru = torch.nn.GRU(input_size=2, hidden_size=2, num_layers=4, batch_first=True)

        self.fc1 = torch.nn.Linear(in_features=4096, out_features=2048)
        # self.fc2 = torch.nn.Linear(in_features=2048, out_features=2048)
        # self.fc3 = torch.nn.Linear(in_features=2048, out_features=2048)
        # self.out = torch.nn.Linear(in_features=2048, out_features=2048)

    def forward(self, x):
        if len(x.shape) > 3: x = x.squeeze()
        x = x.float()
        x = torch.matmul(x, torch.permute(x, (0, 2, 1)))
        x = x.unsqueeze(1)
        x = self.bn2d(x)
        x = self.con1(x)
        x = torch.nn.Dropout(self.dropout)(x)
        x = F.relu(x)
        x = self.con2(x)
        x = torch.nn.Dropout(self.dropout)(x)
        x = F.relu(x)
        x = self.con3(x)
        x = torch.nn.Dropout(self.dropout)(x)
        x = F.relu(x)
        x = self.con4(x)
        x = torch.nn.Dropout(self.dropout)(x)
        x = F.relu(x)
        x = self.fl(x)

        x = self.fc(x)
        x = torch.nn.Dropout(self.dropout)(x)
        x = F.relu(x)

        x = torch.fft.fft(x.float())
        x = x.unsqueeze(1)
        x = torch.cat((torch.real(x), torch.imag(x)), dim=1)

        x = self.bn1d(x)
        x = torch.permute(x, (0, 2, 1))

        x, hn = self.gru(x)
        x = F.relu(x)
        x = self.fl(x)

        x = self.fc1(x)
        # x = torch.nn.Dropout(self.dropout)(x)
        # x = F.relu(x)
        # x = self.fc2(x)
        # x = torch.nn.Dropout(self.dropout)(x)
        # x = F.relu(x)
        # x = self.fc3(x)
        # x = torch.nn.Dropout(self.dropout)(x)
        # x = F.relu(x)
        # x = self.out(x)

        return x


#**************************************************************************************************#
#                                             Class CNN                                            #
#**************************************************************************************************#
#                                                                                                  #
# A simple convolutional neural network (CNN), followed by dense layers.                           #
#                                                                                                  #
#**************************************************************************************************#
class CovCNNalt(torch.nn.Module):
    def __init__(self, dropout=0):
        super(CovCNNalt, self).__init__()
        self.dropout = dropout

        self.bn1d = torch.nn.BatchNorm2d(num_features=1)
        self.fl = torch.nn.Flatten()

        self.con1 = torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(7, 7),
                                    stride=(3, 3), )#padding=(2, 2), dilation=(2, 2))
        self.con2 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(7, 7),
                                    stride=(3, 3), )#padding=(2, 2), dilation=(2, 2))
        self.con3 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(7, 7),
                                    stride=(3, 3), )#padding=(2, 2), dilation=(2, 2))
        self.con4 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5),
                                    stride=(2, 2), )#padding=(2, 2), dilation=(2, 2))
        self.con5 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5),
                                    stride=(2, 2), )#padding=(2, 2), dilation=(2, 2))
        self.con6 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 5),
                                    stride=(2, 2), )#padding=(2, 2), dilation=(2, 2))
        self.con7 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3),
                                    stride=(1, 1), )#padding=(2, 2), dilation=(2, 2))
        self.con8 = torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3),
                                    stride=(1, 1), )#padding=(2, 2), dilation=(2, 2))
        self.con9 = torch.nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=(2, 2),
                                    stride=(1, 1), )#padding=(2, 2), dilation=(2, 2))

        # self.fc1 = torch.nn.Linear(in_features=2048, out_features=2048)
        # self.fc2 = torch.nn.Linear(in_features=2048, out_features=2048)
        # self.fc3 = torch.nn.Linear(in_features=2048, out_features=2048)
        self.out = torch.nn.Linear(in_features=2048, out_features=2048)

    def forward(self, x):
        if len(x.shape) > 3: x = x.squeeze()
        x = x.float()
        x = torch.matmul(x, torch.permute(x, (0, 2, 1)))
        x = x.unsqueeze(1)
        x = self.bn1d(x)
        x = self.con1(x)
        x = torch.nn.Dropout(self.dropout)(x)
        x = F.relu(x)
        x = self.con2(x)
        x = torch.nn.Dropout(self.dropout)(x)
        x = F.relu(x)
        x = self.con3(x)
        x = torch.nn.Dropout(self.dropout)(x)
        x = F.relu(x)
        x = self.con4(x)
        x = torch.nn.Dropout(self.dropout)(x)
        x = F.relu(x)
        x = self.con5(x)
        x = torch.nn.Dropout(self.dropout)(x)
        x = F.relu(x)
        x = self.con6(x)
        x = torch.nn.Dropout(self.dropout)(x)
        x = F.relu(x)
        x = self.con7(x)
        x = torch.nn.Dropout(self.dropout)(x)
        x = F.relu(x)
        x = self.con8(x)
        x = torch.nn.Dropout(self.dropout)(x)
        x = F.relu(x)
        x = self.con9(x)
        x = torch.nn.Dropout(self.dropout)(x)
        x = F.relu(x)
        x = self.fl(x)
        # x = self.fc1(x)
        # x = torch.nn.Dropout(self.dropout)(x)
        # x = F.relu(x)
        # x = self.fc2(x)
        # x = torch.nn.Dropout(self.dropout)(x)
        # x = F.relu(x)
        # x = self.fc3(x)
        # x = torch.nn.Dropout(self.dropout)(x)
        # x = F.relu(x)
        x = self.out(x)
        return x


#**************************************************************************************************#
#                                             Class CNN                                            #
#**************************************************************************************************#
#                                                                                                  #
# A simple convolutional neural network (CNN), followed by dense layers.                           #
# Submission track 2 and 3.                                                                        #
#                                                                                                  #
#**************************************************************************************************#
class CovCandRNN(torch.nn.Module):
    def __init__(self, dropout=0):
        super(CovCandRNN, self).__init__()
        self.dropout = dropout

        self.bn2d = torch.nn.BatchNorm2d(num_features=1)
        self.bn1d = torch.nn.BatchNorm1d(num_features=2)
        self.fl = torch.nn.Flatten()

        self.con1 = torch.nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(7, 7),
                                    stride=(3, 3), )#padding=(2, 2), dilation=(2, 2))
        self.con2 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(7, 7),
                                    stride=(3, 3), )#padding=(2, 2), dilation=(2, 2))
        self.con3 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(7, 7),
                                    stride=(3, 3), )#padding=(2, 2), dilation=(2, 2))
        self.con4 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5),
                                    stride=(2, 2), )#padding=(2, 2), dilation=(2, 2))
        self.con5 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5),
                                    stride=(2, 2), )#padding=(2, 2), dilation=(2, 2))
        self.con6 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 5),
                                    stride=(2, 2), )#padding=(2, 2), dilation=(2, 2))
        self.con7 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3),
                                    stride=(1, 1), )#padding=(2, 2), dilation=(2, 2))
        self.con8 = torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3),
                                    stride=(1, 1), )#padding=(2, 2), dilation=(2, 2))
        self.con9 = torch.nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=(2, 2),
                                    stride=(1, 1), )#padding=(2, 2), dilation=(2, 2))
        # self.fc = torch.nn.Linear(in_features=2048, out_features=2048)

        self.gru = torch.nn.GRU(input_size=2, hidden_size=2, num_layers=4, batch_first=True)

        self.fc1 = torch.nn.Linear(in_features=4096, out_features=2048)
        # self.fc2 = torch.nn.Linear(in_features=2048, out_features=2048)
        # self.fc3 = torch.nn.Linear(in_features=2048, out_features=2048)
        # self.out = torch.nn.Linear(in_features=2048, out_features=2048)

    def forward(self, x):
        if len(x.shape) > 3: x = x.squeeze()
        x = x.float()
        x = torch.matmul(x, torch.permute(x, (0, 2, 1)))
        x = x.unsqueeze(1)
        x = self.bn2d(x)

        x = self.con1(x)
        x = torch.nn.Dropout(self.dropout)(x)
        x = F.relu(x)
        x = self.con2(x)
        x = torch.nn.Dropout(self.dropout)(x)
        x = F.relu(x)
        x = self.con3(x)
        x = torch.nn.Dropout(self.dropout)(x)
        x = F.relu(x)
        x = self.con4(x)
        x = torch.nn.Dropout(self.dropout)(x)
        x = F.relu(x)
        x = self.con5(x)
        x = torch.nn.Dropout(self.dropout)(x)
        x = F.relu(x)
        x = self.con6(x)
        x = torch.nn.Dropout(self.dropout)(x)
        x = F.relu(x)
        x = self.con7(x)
        x = torch.nn.Dropout(self.dropout)(x)
        x = F.relu(x)
        x = self.con8(x)
        x = torch.nn.Dropout(self.dropout)(x)
        x = F.relu(x)
        x = self.con9(x)
        x = torch.nn.Dropout(self.dropout)(x)
        x = F.relu(x)

        x = x.squeeze(-1)
        x = torch.permute(x, (0, 2, 1))

        # x = self.fc(x)
        # x = torch.nn.Dropout(self.dropout)(x)
        # x = F.relu(x)

        x = torch.fft.fft(x.float())
        x = torch.cat((torch.real(x), torch.imag(x)), dim=1)
        x = self.bn1d(x)
        x = torch.permute(x, (0, 2, 1))

        x, hn = self.gru(x)
        x = F.relu(x)
        x = self.fl(x)

        x = self.fc1(x)
        # x = torch.nn.Dropout(self.dropout)(x)
        # x = F.relu(x)
        # x = self.fc2(x)
        # x = torch.nn.Dropout(self.dropout)(x)
        # x = F.relu(x)
        # x = self.fc3(x)
        # x = torch.nn.Dropout(self.dropout)(x)
        # x = F.relu(x)
        # x = self.out(x)

        return x


#**************************************************************************************************#
#                                             Class CNN                                            #
#**************************************************************************************************#
#                                                                                                  #
# A simple convolutional neural network (CNN), followed by dense layers.                           #
#                                                                                                  #
#**************************************************************************************************#
class CovCandRNNComplex(torch.nn.Module):
    def __init__(self, dropout=0):
        super(CovCandRNNComplex, self).__init__()
        self.dropout = dropout

        self.bn2d = torch.nn.BatchNorm2d(num_features=2)
        self.bn1d = torch.nn.BatchNorm1d(num_features=2)
        self.fl = torch.nn.Flatten()

        self.con1 = torch.nn.Conv2d(in_channels=2, out_channels=8, kernel_size=(7, 7),
                                    stride=(3, 3), )#padding=(2, 2), dilation=(2, 2))
        self.con2 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(7, 7),
                                    stride=(3, 3), )#padding=(2, 2), dilation=(2, 2))
        self.con3 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(7, 7),
                                    stride=(3, 3), )#padding=(2, 2), dilation=(2, 2))
        self.con4 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5),
                                    stride=(2, 2), )#padding=(2, 2), dilation=(2, 2))
        self.con5 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5),
                                    stride=(2, 2), )#padding=(2, 2), dilation=(2, 2))
        self.con6 = torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(5, 5),
                                    stride=(2, 2), )#padding=(2, 2), dilation=(2, 2))
        self.con7 = torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3, 3),
                                    stride=(1, 1), )#padding=(2, 2), dilation=(2, 2))
        self.con8 = torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3),
                                    stride=(1, 1), )#padding=(2, 2), dilation=(2, 2))
        self.con9 = torch.nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=(1, 2),
                                    stride=(1, 1), )#padding=(2, 2), dilation=(2, 2))
        self.fc = torch.nn.Linear(in_features=2048, out_features=2048)

        self.gru = torch.nn.GRU(input_size=2, hidden_size=2, num_layers=4, batch_first=True)

        self.fc1 = torch.nn.Linear(in_features=4096, out_features=2048)
        self.fc2 = torch.nn.Linear(in_features=2048, out_features=2048)
        self.fc3 = torch.nn.Linear(in_features=2048, out_features=2048)
        self.out = torch.nn.Linear(in_features=2048, out_features=2048)

    def forward(self, x):
        if len(x.shape) > 3: x = x.squeeze()

        x = torch.matmul(x, torch.permute(x, (0, 2, 1)))
        x = x.unsqueeze(1)

        # from complex to real and imag stacked
        x = torch.cat((torch.real(x), torch.imag(x)), dim=1)
        x = x.float()

        x = self.bn2d(x)
        x = self.con1(x)
        x = torch.nn.Dropout(self.dropout)(x)
        x = F.relu(x)
        x = self.con2(x)
        x = torch.nn.Dropout(self.dropout)(x)
        x = F.relu(x)
        x = self.con3(x)
        x = torch.nn.Dropout(self.dropout)(x)
        x = F.relu(x)
        x = self.con4(x)
        x = torch.nn.Dropout(self.dropout)(x)
        x = F.relu(x)
        x = self.con5(x)
        x = torch.nn.Dropout(self.dropout)(x)
        x = F.relu(x)
        x = self.con6(x)
        x = torch.nn.Dropout(self.dropout)(x)
        x = F.relu(x)
        x = self.con7(x)
        x = torch.nn.Dropout(self.dropout)(x)
        x = F.relu(x)
        x = self.con8(x)
        x = torch.nn.Dropout(self.dropout)(x)
        x = F.relu(x)
        x = self.con9(x)
        x = torch.nn.Dropout(self.dropout)(x)
        x = F.relu(x)

        x = x.squeeze(-1)
        x = torch.permute(x, (0, 2, 1))

        x = self.fc(x)
        x = torch.nn.Dropout(self.dropout)(x)
        x = F.relu(x)

        # from real and imag stacked to complex
        x = torch.complex(x[:, 0], x[:, 1])

        x = torch.fft.fft(x)
        x = x.unsqueeze(1)
        x = torch.cat((torch.real(x), torch.imag(x)), dim=1)

        x = self.bn1d(x)
        x = torch.permute(x, (0, 2, 1))

        x, hn = self.gru(x)
        x = F.relu(x)
        x = self.fl(x)

        x = self.fc1(x)
        x = torch.nn.Dropout(self.dropout)(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = torch.nn.Dropout(self.dropout)(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = torch.nn.Dropout(self.dropout)(x)
        x = F.relu(x)
        x = self.out(x)

        return x