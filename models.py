import yaml

from modules import ViT


def from_template(name, **kwargs):
    with open('hyperparams.json') as stream:
        params = yaml.full_load(stream)[name]
    params.update(kwargs)
    return ViT(**params)
