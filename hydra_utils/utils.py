import importlib
from omegaconf import OmegaConf, DictConfig, ListConfig

def import_class(path: str):
    module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def instantiate_tree(cfg):
    if isinstance(cfg, DictConfig) or isinstance(cfg, dict):
        cfg = OmegaConf.to_container(cfg, resolve=True)

    return _instantiate(cfg)

def _instantiate(cfg):
    if isinstance(cfg, dict):
        if "_target_" in cfg:
            cls = import_class(cfg["_target_"])
            kwargs = {
                key: _instantiate(value)
                for key, value in cfg.items()
                if key != "_target_"
            }
            return cls(**kwargs)

        else:
            return {key: _instantiate(value) for key, value in cfg.items()}

    elif isinstance(cfg, list):
        return [_instantiate(item) for item in cfg]

    else:
        return cfg
