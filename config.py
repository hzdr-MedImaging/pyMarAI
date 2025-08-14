import yaml
import paramiko
import logging
from typing import Optional, Tuple

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
    def get_best_available_host(self) -> Tuple[Optional[str], Optional[str]]:
        machines_data = self.config.get('machines', {})

        if isinstance(machines_data, dict):
            machines_list = [{k: v} for k, v in machines_data.items()]
        elif isinstance(machines_data, list):
            machines_list = machines_data
        else:
            logger.error("Invalid format for 'machines' configuration.")
            return None, None

        for entry in machines_list:
            for hostname, host_cfg in entry.items():
                ssh_client = paramiko.SSHClient()
                ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

                try:
                    ssh_client.connect(hostname)

                    # Check CPU load
                    cpu_threshold = float(host_cfg.get("cpu_threshold", 1.0))

                    # remote command to get CPU load
                    stdin, stdout, stderr = ssh_client.exec_command("cut -d' ' -f1 /proc/loadavg")
                    cpu_load_str = stdout.read().decode().strip()

                    stdout.close()
                    stdin.close()
                    stderr.close()

                    cpu_load = float(cpu_load_str)

                    if cpu_load > cpu_threshold:
                        logger.info(f"{hostname} skipped: CPU load {cpu_load:.2f} > {cpu_threshold}")
                        ssh_client.close()
                        continue

                    # GPU host
                    if host_cfg.get("type") == "gpu":
                        gpu_threshold = int(host_cfg.get("gpu_threshold", 90))  # default 90%

                        # remote command to get GPU info
                        stdin, stdout, stderr = ssh_client.exec_command(
                            "/usr/bin/nvidia-smi --format=csv,noheader --query-gpu=index,utilization.gpu"
                        )
                        gpu_info_str = stdout.read().decode().strip()

                        stdout.close()
                        stdin.close()
                        stderr.close()

                        gpu_info_lines = gpu_info_str.splitlines()

                        free_gpus = []
                        for line in gpu_info_lines:
                            try:
                                index, utilization_str = [part.strip() for part in line.split(",")]
                                utilization = int(utilization_str.replace("%", ""))
                                if utilization <= gpu_threshold:
                                    free_gpus.append(index)
                            except (ValueError, IndexError) as e:
                                logger.warning(f"Failed to parse GPU info line on {hostname}: '{line}' - {e}")
                                continue

                        if free_gpus:
                            logger.info(f"Selected GPU {free_gpus[0]} on {hostname}")
                            ssh_client.close()
                            return hostname, free_gpus[0]
                        else:
                            logger.info(f"{hostname} skipped: all GPUs busy")
                            ssh_client.close()
                            continue

                    # CPU-only host
                    logger.info(f"Selected CPU-only host {hostname}")
                    ssh_client.close()
                    return hostname, None

                except paramiko.AuthenticationException:
                    logger.error(f"Authentication failed for {hostname}.")
                except paramiko.SSHException as e:
                    logger.error(f"SSH error on {hostname}: {e}")
                except Exception as e:
                    logger.error(f"General error checking {hostname}: {e}")
                finally:
                    ssh_client.close()

        logger.warning("No available host found")
        return None, None