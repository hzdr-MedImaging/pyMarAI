import yaml
import paramiko
import logging
from typing import Optional, Tuple
from threading import Lock

logger = logging.getLogger(__name__)

class Singleton(type):
    """
    This is a thread-safe implementation of Singleton.
    """

    _instances = {}
    _lock: Lock = Lock()

    """
    We now have a lock object that will be used to synchronize threads during
    first access to the Singleton.
    """

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        # Now, imagine that the program has just been launched. Since there's no
        # Singleton instance yet, multiple threads can simultaneously pass the
        # previous conditional and reach this point almost at the same time. The
        # first of them will acquire lock and will proceed further, while the
        # rest will wait here.
        with cls._lock:
            # The first thread to acquire the lock, reaches this conditional,
            # goes inside and creates the Singleton instance. Once it leaves the
            # lock block, a thread that might have been waiting for the lock
            # release may then enter this section. But since the Singleton field
            # is already initialized, the thread won't create a new object.
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]

class AppConfig(metaclass=Singleton):

    def __init__(self, path="/usr/local/etc/pymarai.yml"):
        with open(path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.microscopes = self.config.get("microscopes", [])
        self.default_microscope = self.config.get("default_microscope", "-")
        self.machines = self.config.get("machines", [])
        self.utils = self.config.get("utils", {})
        self.nnunet = self.config.get("nnunet", {})

    def get_microscopes(self):
        return self.microscopes

    def get_default_microscope(self):
        return self.default_microscope

    def get_utils(self):
        return self.utils

    def get_machines(self):
        return self.machines

    def get_nnunet(self):
        return self.nnunet

    # dynamically choose the first available machine based on CPU/GPU load
    def get_best_available_host(self, username, password=None, ssh_keys=[]) -> Tuple[Optional[str], Optional[str]]:
        machines_data = self.machines

        if isinstance(machines_data, dict):
            machines_list = [{k: v} for k, v in machines_data.items()]
        elif isinstance(machines_data, list):
            machines_list = machines_data
        else:
            logger.error("Invalid format for 'machines' configuration.")
            return None, None

        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        for entry in machines_list:
            for hostname, host_cfg in entry.items():
                try:
                    if len(ssh_keys) > 0:
                        ssh.connect(hostname, port=22, username=username, key_filename=ssh_keys[0])
                    else:
                        ssh.connect(hostname, port=22, username=username, password=password)

                    # Check CPU load
                    cpu_threshold = float(host_cfg.get("cpu_threshold", 1.0))

                    # remote command to get CPU load
                    stdin, stdout, stderr = ssh.exec_command("cut -d' ' -f1 /proc/loadavg")
                    cpu_load_str = stdout.read().decode().strip()

                    stdout.close()
                    stdin.close()
                    stderr.close()

                    cpu_load = float(cpu_load_str)

                    if cpu_load > cpu_threshold:
                        logger.info(f"{hostname} skipped: CPU load {cpu_load:.2f} > {cpu_threshold}")
                        ssh.close()
                        continue

                    # GPU host
                    if host_cfg.get("type") == "gpu":
                        gpu_threshold = int(host_cfg.get("gpu_threshold", 90))  # default 90%

                        # remote command to get GPU info
                        stdin, stdout, stderr = ssh.exec_command(
                            f"{self.utils['nvidia-smi']} --format=csv,noheader --query-gpu=index,utilization.gpu"
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
                            ssh.close()
                            return hostname, free_gpus[0]
                        else:
                            logger.info(f"{hostname} skipped: all GPUs busy")
                            ssh.close()
                            continue

                    # CPU-only host
                    logger.info(f"Selected CPU-only host {hostname}")
                    ssh.close()
                    return hostname, None

                except paramiko.AuthenticationException:
                    logger.error(f"Authentication failed for {hostname}.")
                except paramiko.SSHException as e:
                    logger.error(f"SSH error on {hostname}: {e}")
                except Exception as e:
                    logger.error(f"General error checking {hostname}: {e}")
                finally:
                    ssh.close()

        logger.warning("No available host found")
        return None, None
