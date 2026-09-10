from logging import getLogger

import numpy as np
import torch.nn as nn


class SocialRecommender(nn.Module):
    def __init__(self, config, dataset):
        self.logger = getLogger(__name__)
        super().__init__()
        self.USER_ID = config.USER_ID_FIELD
        self.ITEM_ID = config.ITEM_ID_FIELD
        self.NEG_ITEM_ID = config.NEG_PREFIX + self.ITEM_ID
        self.n_users = dataset.num(self.USER_ID)
        self.n_items = dataset.num(self.ITEM_ID)
        self.device = config.device

    def other_parameter(self):
        if hasattr(self, "other_parameter_name"):
            return {key: getattr(self, key) for key in self.other_parameter_name}
        return {}

    def load_other_parameter(self, para):
        if para is None:
            return
        for key, value in para.items():
            setattr(self, key, value)

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum(np.prod(p.size()) for p in model_parameters)
        return super().__str__() + f"\nTrainable parameters: {params}"
