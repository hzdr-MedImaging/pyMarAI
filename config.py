import os
import yaml

class AppConfig:
    def __init__(self, path=None):
        if path is None:
            if os.name == 'nt':
                path = "C:\\pymarai\\pymarai.yml"
            else:
                path = os.path.expanduser("/usr/local/etc/pymarai.yml")

        with open(path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.microscopes = self.config.get("microscopes", [])
        self.default_microscope = self.config.get("default_microscope", "-")
        self.machines = self.config.get("machines", {})

    def get_microscopes(self):
        return self.config.get("microscopes", [])

    def get_default_microscope(self):
        return self.default_microscope
