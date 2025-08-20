import os
import shutil
import subprocess
import platform
import yaml
import paramiko
import threading
import logging
import shlex
import re

from pymarai.config import AppConfig

from abc import ABC

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logging.getLogger('paramiko').setLevel(logging.WARNING)
logging.getLogger('paramiko.transport').setLevel(logging.WARNING)

class UserCancelledError(Exception):
    # custom exception for user-initiated cancellation
    pass

class MarAiBase(ABC):
    def __init__(self, hostname=platform.node(), stop_event=None, gpu_id=None):

        # get config sections
        config = AppConfig()
        self.utils = config.get_utils()
        self.machines = config.get_machines()
        self.nnunet_cfg = config.get_nnunet()

        self.nnunet_v_output_dir = None
        self.nnunet_results_folder = None
        self.nnunet_model_name = None
        self.nnunet_task_name = None
        self.roi2rdf_binary_path = None
        self.roi2rdf_config_path = None
        self.base_temp_dir = None
        self.temp_inference_dir = None

        self.progress_callback = None
        self.processed_files_count = 0
        self.total_expected_files = 0
        self.reported_nnunet_files = set()
        self.reported_rdf_files = set()

        # stop event from the GUI's multiprocessing.Process
        # allows MarAi to check if it should stop
        self.stop_event = stop_event

        # get paths for mic2ecat and roi2rdf
        try:
            self.mic2ecat_path = self.utils['mic2ecat']
            self.roi2rdf_path = self.utils['roi2Rdf']
            self.conda_path = self.utils['conda']
            self.nnunet_predict_path = self.utils['nnunet_predict']
            self.nnunet_train_path = self.utils['nnunet_train']
        except KeyError as e:
            raise RuntimeError(f"[ERROR] Missing required path in config: {e}.\n")

        # Event to signal when nnUNet process itself has finished
        self.nnunet_process_finished_event = threading.Event()

        # find matching host config
        selected_host_cfg = self._find_host_config(self.machines, hostname)
        if not selected_host_cfg:
            raise ValueError("[ERROR] No matching remote device found in config.\n")

        # extract details from config
        self.host_cfg = selected_host_cfg
        self.gpu_id = gpu_id

    # find device config by hostname or use default
    def _find_host_config(self, machines_data, hostname=None):
        logger.debug(f"[_find_host_config] machines_data type: {type(machines_data)}, content: {machines_data}")

        if isinstance(machines_data, dict):
            machines_list = [{k: v} for k, v in machines_data.items()]
        elif isinstance(machines_data, list):
            machines_list = machines_data
        else:
            raise ValueError(
                f"[ERROR] Invalid format for 'machines' in config: expected dict or list, got {type(machines_data)}")

        for entry in machines_list:
            if not isinstance(entry, dict):
                continue
            host_key = list(entry.keys())[0]
            cfg = entry[host_key]
            full_cfg = {'hostname': host_key}
            full_cfg.update(cfg)
            if hostname and hostname == host_key:
                return full_cfg
            if cfg.get("default", False) and hostname is None:
                return full_cfg

        return None

    # build base nnunet command
    def get_nnunet_base_cmd(self, input_dir, output_dir):
        env = self.nnunet_cfg.get('env', 'nnunet')
        dataset = self.nnunet_cfg.get('dataset')
        trainer = self.nnunet_cfg.get('trainer')
        config = self.nnunet_cfg.get('config')
        plans = self.nnunet_cfg.get('plans')
        folds = self.nnunet_cfg.get('folds', [0])

        if not all([dataset, trainer, config, plans]):
            raise ValueError("[ERROR] Missing one or more required nnUNet config fields.\n")

        folds_str = ' '.join(map(str, folds))

        return (
            f"{self.conda_path} run -n {env} --live-stream {self.nnunet_predict_path} "
            f"-d {dataset} -i {input_dir} -o {output_dir} "
            f"-f {folds_str} -tr {trainer} -c {config} -p {plans}"
        )

    # decide whether to add -device cpu
    def _get_device_flag(self, host_cfg):
        if host_cfg and isinstance(host_cfg, dict):
            type = host_cfg.get("type", "cpu")
            if type == 'cpu':
                return "-device cpu"
        return ""

    # Modified runCommand to accept progress_pattern and progress_callback
    def runCommand(self, cmd, stream_output=False, process_event_on_completion=None,
                   progress_pattern=None, original_input_files_map=None):
        raise NotImplementedError

    # run the nnunet prediction workflow
    def predictCall(self, input_files, microscope_number, output_dir, progress_callback=None):
        self.progress_callback = progress_callback
        self.total_expected_files = len(input_files)
        self.processed_files_count = 0
        self.reported_nnunet_files.clear()
        self.reported_rdf_files.clear()
        self.nnunet_process_finished_event.clear()

        if self.stop_event and self.stop_event.is_set():
            logger.info("Prediction aborted before start.")
            return

        tempDir = os.path.join(output_dir, f"tmp-{os.getpid()}-{threading.get_ident()}")
        os.makedirs(tempDir, exist_ok=True)
        nnunet_input_dir = os.path.join(tempDir, "nnunet_input")
        os.makedirs(nnunet_input_dir, exist_ok=True)
        nnunet_output_dir = os.path.join(tempDir, "nnunet_output")
        os.makedirs(nnunet_output_dir, exist_ok=True)

        original_input_files_map = {self.get_file_prefix(f): f for f in input_files}

        # create soft links in our tempDir which will point to the
        # correct original tif files just two directories lower than tempDir
        logger.info("Soft linking input files to temporary directory...")
        for input_file in input_files:
            os.symlink(os.path.join("..", "..", os.path.basename(input_file)), os.path.join(tempDir, os.path.basename(input_file)))

        # --- Run mic2ecat ---
        logger.info("Running mic2ecat...")
        mic2ecat_cmd = f"cd {tempDir} && find . -maxdepth 1 -name '*.tif' -or -name '*.png' | xargs {self.mic2ecat_path} -j {microscope_number} -v"
        self.runCommand(mic2ecat_cmd, stream_output=True)

        # --- Symlink mic2ecat output in nnunet_input_dir ---
        logger.info("Soft linking mic2ecat output to nnunet_input_dir...")
        for filename in os.listdir(tempDir):
            old_path = os.path.join("..", filename)
            if filename.endswith('.v'):
                base = os.path.basename(filename)[:-2]
                new_filename = f"{base}_0000.v"
            else:
                continue
            os.symlink(old_path, os.path.join(nnunet_input_dir, new_filename))
            logger.debug(f"Symlink {os.path.join(nnunet_input_dir, new_filename)} → {old_path}")

        # --- Run nnUNet ---
        nnunet_cmd = self.get_nnunet_base_cmd(nnunet_input_dir, nnunet_output_dir)
        device_flag = self._get_device_flag(self.host_cfg)
        nnunet_cmd_full = f"{nnunet_cmd} {device_flag}"
        nnunet_progress_pattern = r"done with (\S+)"
        nnunet_thread = threading.Thread(target=self.runCommand,
                                         args=(nnunet_cmd_full, True, self.nnunet_process_finished_event,
                                               nnunet_progress_pattern, original_input_files_map))
        logger.info(f"Executing nn-UNet: '{nnunet_cmd_full}'")
        nnunet_thread.start()
        nnunet_thread.join()

        # --- Run roi2rdf ---
        roi2rdf_cmd = f"cd {nnunet_output_dir} && {self.roi2rdf_path} -v *.v"
        self.runCommand(roi2rdf_cmd, stream_output=True)

        # --- Cleanup output_dir stuff from previous runs before storing new stuff
        for filename_base in original_input_files_map:
            filename_base = f"{filename_base}_m{microscope_number}"
            for filename in os.listdir(output_dir):
                if filename.startswith(filename_base):
                    os.remove(os.path.join(output_dir, filename))

        # --- Move raw _0000.v from tempDir to output_dir ---
        logger.info("Moving raw mic2ecat outputs from tempDir to output_dir...")
        for filename in os.listdir(tempDir):
            old_path = os.path.join(tempDir, filename)
            if filename.endswith('.v'):
                base = os.path.basename(filename)[:-2]
                new_filename = f"{base}_m{microscope_number}.v"
            else:
                continue

            try:
              logger.debug(f"Moving {old_path} → {os.path.join(output_dir, new_filename)}")
              os.rename(old_path, os.path.join(output_dir, new_filename))
            except Exception as e:
              logger.error(f"Could not move {old_path} → {os.path.join(output_dir, new_filename)}: {e}")

        # --- Move .rdf from nnunet_output_dir to output_dir and .v as _cnn.v ---
        logger.info("Moving roi2rdf outputs from nnunet_output to output_dir...")
        for filename in os.listdir(nnunet_output_dir):
            old_path = os.path.join(nnunet_output_dir, filename)
            if filename.endswith('.rdf'):
                base = os.path.basename(filename)[:-4]
                new_filename = f"{base}_m{microscope_number}.rdf"
            elif filename.endswith('.v'):
                base = os.path.basename(filename)[:-2]
                new_filename = f"{base}_m{microscope_number}_cnn.v"
            else:
                continue

            try:
              logger.debug(f"Moving {old_path} → {os.path.join(output_dir, new_filename)}")
              os.rename(old_path, os.path.join(output_dir, new_filename))
            except Exception as e:
              logger.error(f"Could not move {old_path} → {os.path.join(output_dir, new_filename)}: {e}")

        # Cleanup
        shutil.rmtree(tempDir)

    # returns the base filename without extension
    def get_file_prefix(self, input_file):
        return os.path.splitext(os.path.basename(input_file))[0]

    # returns the expected filename for nnUNet input in its temp dir
    def _get_nnunet_input_name_for_temp_copy(self, file_prefix):
        return f"{file_prefix}_0000.v"

    # returns the expected filename for nnUNet output within its temp dir
    def _get_nnunet_output_name_in_temp(self, file_prefix):
        return f"{file_prefix}.v"

    # returns the expected filename for roi2rdf output within its temp dir
    def _get_roi2rdf_output_name_in_temp(self, file_prefix):
        return f"{file_prefix}.rdf"

    # helper to get expected full paths for files within the nnunet_output_dir
    def _get_expected_full_paths_in_nnunet_output_dir(self, nnunet_output_dir, file_prefix):
        nnunet_v_path_in_temp = os.path.join(nnunet_output_dir, self._get_nnunet_output_name_in_temp(file_prefix))
        rdf_path_in_temp = os.path.join(nnunet_output_dir, self._get_roi2rdf_output_name_in_temp(file_prefix))
        return nnunet_v_path_in_temp, rdf_path_in_temp

class MarAiLocal(MarAiBase):
    def __init__(self, stop_event=None, gpu_id=None):
        super().__init__(hostname=platform.node(), stop_event=stop_event, gpu_id=gpu_id)

    # run a local shell command
    def runCommand(self, cmd, stream_output=False, process_event_on_completion=None,
                   progress_pattern=None, original_input_files_map=None):
        if isinstance(cmd, list):
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, env=os.environ.copy())
        else:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, env=os.environ.copy(),
                                       shell=True, executable="/bin/bash" if platform.system() != 'Windows' else None)

        if stream_output:
            for line in process.stdout:
                print(line, end='')
                if progress_pattern and self.progress_callback and original_input_files_map:
                    match = re.search(progress_pattern, line)
                    if match:
                        filename = match.group(1)
                        file_prefix_from_output = os.path.splitext(filename)[0]

                        if file_prefix_from_output in original_input_files_map and \
                           file_prefix_from_output not in self.reported_nnunet_files:
                            self.reported_nnunet_files.add(file_prefix_from_output)
                            current_nnunet_count = len(self.reported_nnunet_files)
                            original_input_filepath = original_input_files_map[file_prefix_from_output]

                            self.progress_callback(
                                current_nnunet_count,
                                self.total_expected_files,
                                os.path.basename(original_input_filepath),
                                "Running prediction"
                            )
                            logger.info(f"Reported nnUNet progress for {file_prefix_from_output} via stdout: {current_nnunet_count}/{self.total_expected_files}")
        else:
            output, _ = process.communicate()
            print(output)

        exit_status = process.wait()
        if exit_status != 0:
            raise RuntimeError(f"[ERROR] Command failed with exit status {exit_status}: {cmd}.\n")

        if process_event_on_completion:
            process_event_on_completion.set()  # signal that this command has completed

class MarAiRemote(MarAiBase):
    def __init__(self, hostname=None, username=None, password=None, ssh_keys=None, stop_event=None, gpu_id=None):
        super().__init__(hostname=hostname, stop_event=stop_event, gpu_id=gpu_id)

        # get remote host details from host config
        self.hostname = self.host_cfg.get('hostname')
        self.ipaddress = self.host_cfg.get('ip')
        self.username = username
        self.password = password

        if not self.hostname:
            raise ValueError("[ERROR] Hostname not specified or found in configuration.\n")

        self.connected = False
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.ssh_keys = ssh_keys

    def __del__(self):
        if hasattr(self, "connected") and self.connected:
            self.disconnect()

    def connect(self):
        if not self.connected:
            logger.info(f"Connecting to remote host {self.hostname}...")
            try:
                if len(self.ssh_keys) > 0:
                    self.ssh.connect(self.hostname, port=22, username=self.username, key_filename=self.ssh_keys[0])
                    logger.info(f"Connection successful via key-based authentication ({self.ssh_keys[0]}).")
                else:
                    self.ssh.connect(self.hostname, port=22, username=self.username, password=self.password)
                    logger.info(f"Connection successful via password authentication")

                self.connected = True
            except Exception as e:
                logger.error(f"Connection failed: {e}")

    def disconnect(self):
        if self.connected:
            if self.ssh: # check if ssh object exists before closing
                try:
                    self.ssh.close()
                except Exception as e:
                    logger.warning(f"[ERROR] Error closing SSH client: {e}")
        self.connected = False

        # reinit ssh client for next connection
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # run command remotely and stream output (now can signal completion)
    def runCommand(self, cmd, stream_output=False, process_event_on_completion=None,
                   progress_pattern=None, original_input_files_map=None):

        # make sure we are connected
        self.connect()

        if not self.connected:
            raise RuntimeError("[ERROR] Not connected to remote host.")

        if isinstance(cmd, list):
            cmd_quoted = [shlex.quote(str(c)) for c in cmd]
            cmd_str = ' '.join(cmd_quoted)
        else:
            cmd_str = str(cmd)

        # add CUDA_VISIBLE_DEVICES for GPU-relevant commands
        gpu_tools = [
            "nnUNetv2_predict",
            "nnUNetv2_train"
        ]

        if self.gpu_id is not None and any(tool in cmd_str for tool in gpu_tools):
            cmd_str = f"CUDA_VISIBLE_DEVICES={self.gpu_id} {cmd_str}"
        full_cmd = f"stdbuf -o0 bash -c {shlex.quote(cmd_str)}"

        stdin, stdout, stderr = self.ssh.exec_command(full_cmd, get_pty=True)

        if stream_output:
            for line in iter(stdout.readline, ""):
                print(line, end='')
                if progress_pattern and self.progress_callback and original_input_files_map:
                    match = re.search(progress_pattern, line)
                    if match:
                        filename = match.group(1)
                        file_prefix_from_output = os.path.splitext(filename)[0]
                        if file_prefix_from_output in original_input_files_map and \
                                file_prefix_from_output not in self.reported_nnunet_files:
                            self.reported_nnunet_files.add(file_prefix_from_output)
                            current_nnunet_count = len(self.reported_nnunet_files)
                            original_input_filepath = original_input_files_map[file_prefix_from_output]
                            self.progress_callback(
                                current_nnunet_count,
                                self.total_expected_files,
                                os.path.basename(original_input_filepath),
                                "Running prediction"
                            )
            for line in iter(stderr.readline, ""):
                print(line, end='')
        else:
            output = stdout.read().decode('utf-8') + stderr.read().decode('utf-8')
            print(output)

        exit_status = stdout.channel.recv_exit_status()
        if exit_status != 0:
            raise RuntimeError(f"[ERROR] Remote command failed with exit status {exit_status}: {cmd_str}.\n")

        if process_event_on_completion:
            process_event_on_completion.set()

    def predictCall(self, input_files, microscope_number, output_dir, progress_callback=None):
        self.connect()
        super().predictCall(input_files, microscope_number, output_dir, progress_callback)
        self.disconnect()
