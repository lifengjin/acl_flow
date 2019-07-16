import hashlib
class HyperParameterCollector:

    def __init__(self):
        self.hyperparameters = {}

    def add_param(self, param_name, param_val):
        self.hyperparameters[param_name] = param_val
        return param_val

    def add_param_list(self, param_name_list, param_val_list):
        for index, name in enumerate(param_name_list):
            self.hyperparameters[name] = param_val_list[index]
        return param_val_list

    def remove_param(self, param_name):
        self.hyperparameters.pop(param_name)

    def write_out(self):
        keys = list(self.hyperparameters.keys())
        keys.sort()
        string = ''
        for key in keys:
            string += '{} : {} \n'.format(key, self.hyperparameters[key])
        string_hash = hashlib.md5(string.encode()).hexdigest()
        string = '#Hyperparam hash# : {}\n'.format(string_hash) + string
        string = '\n---------Hyperparameters:\n' + '<' * 50 + '\n' + string
        string += '>'*50 + '\n'
        return string