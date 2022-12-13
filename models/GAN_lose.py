import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss


class GANLoss(nn.Module):
    def __init__(self, real_label=1.0, fake_label=0.0):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(real_label))
        self.register_buffer('fake_label', torch.tensor(fake_label))
        self.loss = BCEWithLogitsLoss()

    def get_labels(self, predictions, target_is_real):
        if target_is_real:
            labels = self.real_label
        else:
            labels = self.fake_label
        return labels.expand_as(predictions)

    def __call__(self, predictions, target_is_real):
        labels = self.get_labels(predictions, target_is_real)
        loss = self.loss(predictions, labels)
        return loss
