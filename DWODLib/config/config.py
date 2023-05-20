
"""
    Custom dictionary class for config file
"""

class ConfigDict(dict):
    def __init__(self, config_dict):
        super().__init__(config_dict)

    def update(self, new_config_dict):
        for key, value in new_config_dict.items():
            ## if key not in config file, raise error
            if key not in self.keys():
                raise KeyError(f"Key {key} not in config file!")
            else:
                self[key] = value