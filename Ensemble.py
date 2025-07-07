import torch
import torch.nn as nn
import torch.nn.functional as F


class EnsembleModel(nn.Module):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = models
        
        for model in self.models:
            model.eval() 
            for param in model.parameters():
                param.requires_grad = False

    def forward(self, x):
        all_probs = []
        for model in self.models:
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs)
        
        average_probs = torch.mean(torch.stack(all_probs), dim=0)
        return torch.log(average_probs)
