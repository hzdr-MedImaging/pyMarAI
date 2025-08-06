import yaml
import logging

logger = logging.getLogger(__name__)

class AppConfig:
    def __init__(self, path="/usr/local/etc/pymarai.yml"):
        with open(path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.microscopes = self.config.get("microscopes", [])
        self.default_microscope = self.config.get("default_microscope", "-")
        self.machines = self.config.get("machines", {})

    def get_microscopes(self):
        return self.microscopes

    def get_default_microscope(self):
        return self.default_microscope

    def get_default_host(self):
        try:
            machines_data = self.config.get('machines', [])

            # The 'machines' section can be a dictionary or a list of dictionaries.
            # Convert to a consistent list format for iteration.
            if isinstance(machines_data, dict):
                machines_list = [{k: v} for k, v in machines_data.items()]
            elif isinstance(machines_data, list):
                machines_list = machines_data
            else:
                logger.error(
                    f"[ERROR] Invalid format for 'machines' in config: expected dict or list, got {type(machines_data)}")
                return None

            for entry in machines_list:
                if isinstance(entry, dict):
                    for hostname, host_cfg in entry.items():
                        if host_cfg.get("default", False):
                            return hostname
            return None
        except Exception as e:
            logger.error(f"Failed to find default host in config: {e}")
            return None
