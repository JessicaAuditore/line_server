import copy
from .model import Model

__all__ = ['build_model']
support_model = ['Model']


def build_model(config):
    copy_config = copy.deepcopy(config)
    arch_type = copy_config.pop('type')
    arch_model = eval(arch_type)(copy_config)
    return arch_model
