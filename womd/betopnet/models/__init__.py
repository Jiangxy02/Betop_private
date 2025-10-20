'''
Behavioral Topology (BeTop): https://arxiv.org/abs/2409.18031
'''

from .betopnet import BeTopNet
from .wayformer import Wayformer
from .mtr_plus_plus import MTRPP

__all__ = {
    'BeTopNet': BeTopNet,
    'Wayformer':Wayformer,
    'MTRPP': MTRPP,
}


def build_model(config):
    model = __all__[config.NAME](
        config=config
    )
    return model


# Create a module-like object to support "from betopnet.models import model as model_utils"
class ModelModule:
    build_model = staticmethod(build_model)

import sys
sys.modules['betopnet.models.model'] = ModelModule()