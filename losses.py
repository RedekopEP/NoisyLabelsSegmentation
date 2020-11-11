import torch
from torch import nn


class LossBinaryDice(nn.Module):
    def __init__(self, dice_weight: int = 1, activation: bool = False):
        super(LossBinaryDice, self).__init__()
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.activation = activation

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.squeeze().view(-1)
        outputs = outputs.squeeze().view(-1)

        loss = (1 - self.dice_weight) * self.nll_loss(outputs, targets)

        if self.dice_weight:
            smooth = 0.1
            target = (targets > 0.0).float()
            if self.activation:
                prediction = outputs
            else:
                prediction = torch.sigmoid(outputs)

            dice_part = (1 - (2 * torch.sum(prediction * target, dim=0) + smooth) /
                         (torch.sum(prediction, dim=0) + torch.sum(target, dim=0) + smooth))

            loss += self.dice_weight * dice_part.mean()
        return loss
