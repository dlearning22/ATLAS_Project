import sys
from torch import nn
import torch


class DiceLoss(nn.Module):
    """
    Dice loss function class
    """
    def __init__(self, squared_denom=False):
        super(DiceLoss, self).__init__()
        self.smooth = sys.float_info.epsilon
        self.squared_denom = squared_denom

    def forward(self, x, target):
        x = x.view(-1)
        target = target.view(-1)
        intersection = (x * target).sum()
        numer = 2. * intersection + self.smooth
        factor = 2 if self.squared_denom else 1
        denom = x.pow(factor).sum() + target.pow(factor).sum() + self.smooth
        dice_index = numer / denom
        return 1 - dice_index


class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1, alpha=ALPHA, beta=BETA, gamma=GAMMA):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        #inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        ALPHA = 0.5
        BETA = 0.5
        GAMMA = 1
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum() #intesection   
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        FocalTversky = (1 - Tversky)**gamma
                       
        return FocalTversky
