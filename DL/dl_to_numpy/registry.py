class Registry(object):
    def __init__(self):
        self._module_dict = dict()

    def get(self, key, kwargs):
        return self._module_dict.get(key, None)(**kwargs)

    def register_module(self, cls):
        if cls.__name__ in self._module_dict:
            print("ERROR: key already exists")
        else:
            self._module_dict[cls.__name__] = cls
            