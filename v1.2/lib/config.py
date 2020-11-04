import os
import yaml

class Config(dict):
    def __init__(self, config_path):
        with open(config_path, 'r', encoding='UTF-8') as f:
            self._yaml = f.read()
            self._dict = yaml.load(self._yaml)
            self._dict['PATH'] = os.path.dirname(config_path)

    def __getattr__(self, name):
        if self._dict.get(name) is not None:
            return self._dict[name]

        if DEFAULT_CONFIG.get(name) is not None:
            return DEFAULT_CONFIG[name]
        return None

    def print(self):
        print('configurations:')
        print('---------------------------------')
        print(self._yaml)
        print('')
        print('---------------------------------')
        print('')

DEFAULT_CONFIG = {
    'LR_A': 0.001,                     # learning rate for actor
    'LR_C': 0.0001,                    # learning rate for critic
    'GAMMA': 0.5,                      # reward discount
    'TAU': 0.01,                       # soft replacement
    'MEMORY_CAPACITY': 1000,           # memory
    'BATCH_SIZE': 32,                   # batch size
    'GPU': [0]                         # list of gpu ids

}