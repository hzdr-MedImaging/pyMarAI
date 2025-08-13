import yaml
import logging
import subprocess

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

    # dynamically choose the first available machine based on CPU/GPU load
    def get_best_available_host(self):
        machines_data = self.config.get('machines', {})

        if isinstance(machines_data, dict):
            machines_list = [{k: v} for k, v in machines_data.items()]
        elif isinstance(machines_data, list):
            machines_list = machines_data
        else:
            logger.error(f"Invalid format for 'machines'")
            return None, None

        for entry in machines_list:
            for hostname, host_cfg in entry.items():
                try:
                    # check CPU load
                    cpu_threshold = float(host_cfg.get("cpu_threshold", 1.0))
                    cpu_load = float(subprocess.check_output(
                        ["ssh", "-o", "StrictHostKeyChecking=no", hostname,
                         "cut -d' ' -f1 /proc/loadavg"],
                        stderr=subprocess.DEVNULL
                    ).decode().strip())

                    if cpu_load > cpu_threshold:
                        logger.info(f"{hostname} skipped: CPU load {cpu_load} > {cpu_threshold}")
                        continue

                    # GPU host
                    if host_cfg.get("type") == "gpu":
                        gpu_threshold = int(host_cfg.get("gpu_threshold", 90))  # default 90%
                        gpu_info = subprocess.check_output(
                            ["ssh", "-o", "StrictHostKeyChecking=no", hostname,
                             "/usr/bin/nvidia-smi --format=csv,noheader --query-gpu=index,utilization.gpu"],
                            stderr=subprocess.DEVNULL
                        ).decode().strip().splitlines()

                        free_gpus = [
                            line.split(",")[0].strip()
                            for line in gpu_info
                            if int(line.split(",")[1].strip().replace("%", "")) <= gpu_threshold
                        ]

                        if free_gpus:
                            logger.info(f"Selected GPU {free_gpus[0]} on {hostname}")
                            return hostname, free_gpus[0]
                        else:
                            logger.info(f"{hostname} skipped: all GPUs busy")
                            continue  # do not fallback to CPU for a GPU host

                    # CPU-only host
                    logger.info(f"Selected CPU-only host {hostname}")
                    return hostname, None

                except Exception as e:
                    logger.error(f"Error checking {hostname}: {e}")
                    continue

        logger.warning("No available host found")
        return None, None