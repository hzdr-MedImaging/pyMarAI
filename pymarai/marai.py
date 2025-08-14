import os
import shutil
import subprocess
import tarfile
import tempfile
import platform
import yaml
import paramiko
import threading
import time
import logging
import shlex
import re
import io

from abc import ABC

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

logging.getLogger('paramiko').setLevel(logging.WARNING)
logging.getLogger('paramiko.transport').setLevel(logging.WARNING)

class UserCancelledError(Exception):
    # custom exception for user-initiated cancellation
    pass

class MarAiBase(ABC):
    def __init__(self, cfg_path, stop_event=None):
        # load config file
        cfg_path = cfg_path or "/usr/local/etc/pymarai.yml"
        self.cfg = self._load_config(cfg_path)

        self.nnunet_v_output_dir = None
        self.nnunet_results_folder = None
        self.nnunet_model_name = None
        self.nnunet_task_name = None
        self.roi2rdf_binary_path = None
        self.roi2rdf_config_path = None
        self.base_temp_dir = None
        self.temp_inference_dir = None
        self.sftp = None

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
            self.mic2ecat_path = self.cfg['utils']['mic2ecat']
            self.roi2rdf_path = self.cfg['utils']['roi2Rdf']
        except KeyError as e:
            raise RuntimeError(f"[ERROR] Missing required path in config: {e}.\n")

        # get nnunet section from config
        self.nnunet_cfg = self.cfg.get('nnunet', {})
        self.localTempDir = tempfile.TemporaryDirectory(prefix="marai-")
        self.tempId = self.localTempDir.name.split('-')[1]

        # Event to signal when nnUNet process itself has finished
        self.nnunet_process_finished_event = threading.Event()

    # clean up temporary directory
    def __del__(self):
        try:
            self.localTempDir.cleanup()
        except Exception as e:
            logger.warning(f"[ERROR] Error cleaning up local temporary directory {self.localTempDir.name}: {e}")

    # config file loader
    def _load_config(self, cfg_path):
        if not os.path.exists(cfg_path):
            logger.error(f"[ERROR] Config file not found: {cfg_path}.")
            raise FileNotFoundError(f"[ERROR] Config file not found: {cfg_path}.")
        with open(cfg_path, 'r') as f:
            return yaml.safe_load(f)

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
            f"/usr/local/miniforge3/condabin/conda run -n {env} --live-stream nnUNetv2_predict "
            f"-d {dataset} -i {input_dir} -o {output_dir} "
            f"-f {folds_str} -tr {trainer} -c {config} -p {plans}"
        )

    # decide whether to add -device cpu
    def _get_device_flag(self, host_cfg):
        if host_cfg and isinstance(host_cfg, dict):
            cpu = host_cfg.get("cpu", False)
            gpu = host_cfg.get("gpu", True)
            if cpu and not gpu:
                return "-device cpu"
        return ""

    # Modified runCommand to accept progress_pattern and progress_callback
    def runCommand(self, cmd, stream_output=False, process_event_on_completion=None,
                   progress_pattern=None, original_input_files_map=None):
        raise NotImplementedError

    def predictCall(self, input_files, microscope_number, output_dir, progress_callback=None):
        raise NotImplementedError

    def _list_directory(self, directory_path):
        raise NotImplementedError

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

    # wait for a file to exist with a timeout
    def _wait_for_file_existence(self, file_path, timeout=300, check_interval=5):
        start_time = time.time()
        while not self._check_file_existence(file_path):
            if time.time() - start_time > timeout:
                raise TimeoutError(f"[ERROR] Timeout waiting for file: {file_path}")
            time.sleep(check_interval)
        return True

    # abstract method for checking file existence
    def _check_file_existence(self, file_path):
        raise NotImplementedError

    # The _monitor_nnunet_output method has been removed.
    # Progress for nnUNet is now directly reported from runCommand by parsing its stdout.

    def _get_file_size(self, file_path, is_remote):
        if is_remote:
            return self.sftp.stat(file_path).st_size
        else:
            return os.path.getsize(file_path)

class MarAiLocal(MarAiBase):
    def __init__(self, cfg_path=None, stop_event=None):
        super().__init__(cfg_path, stop_event=stop_event)

    # run a local shell command
    def runCommand(self, cmd, stream_output=False, process_event_on_completion=None,
                   progress_pattern=None, original_input_files_map=None):
        if isinstance(cmd, list):
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        else:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True,
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


    def _check_file_existence(self, file_path):
        return os.path.exists(file_path)

    def _list_directory(self, directory_path):
        return os.listdir(directory_path)

    def predictCall(self, input_files, microscope_number, output_dir, progress_callback=None):
        self.progress_callback = progress_callback
        self.total_expected_files = len(input_files)
        self.processed_files_count = 0
        self.reported_nnunet_files.clear()
        self.reported_rdf_files.clear()
        self.nnunet_process_finished_event.clear()

        if self.stop_event and self.stop_event.is_set():
            logger.info("[INFO] Local prediction aborted before start.")
            return

        nnunet_input_dir = os.path.join(self.localTempDir.name, "nnunet_input")
        nnunet_output_dir = os.path.join(self.localTempDir.name, "nnunet_output")
        os.makedirs(nnunet_input_dir, exist_ok=True)
        os.makedirs(nnunet_output_dir, exist_ok=True)

        original_input_files_map = {self.get_file_prefix(f): f for f in input_files}

        # --- Copy input files to temp dir ---
        logger.info("[INFO] Copying input files to temporary directory...")
        for input_file in input_files:
            shutil.copy(input_file, os.path.join(self.localTempDir.name, os.path.basename(input_file)))

        # --- Run mic2ecat  ---
        logger.info("[INFO] Running mic2ecat on all files in temp dir...")
        mic2ecat_cmd = f"cd {self.localTempDir.name} && {self.mic2ecat_path} -j {microscope_number} *.tif"
        self.runCommand(mic2ecat_cmd, stream_output=True)

        # --- Move mic2ecat .v outputs into nnunet_input ---
        logger.info("[INFO] Moving and renaming mic2ecat outputs into nnunet_input directory...")

        move_cmd = (
            f"cd {self.localTempDir.name} && "
            f"for file in *.v; do "
            f"  prefix=$(basename \"$file\" .v); "
            f"  mv -- \"$file\" \"{nnunet_input_dir}/${{prefix}}_0000.v\"; "
            f"done"
        )
        self.runCommand(move_cmd, stream_output=True)

        # --- Run nnUNet ---
        nnunet_cmd = self.get_nnunet_base_cmd(nnunet_input_dir, nnunet_output_dir) + " -device cpu"
        nnunet_progress_pattern = r"done with (\S+)"
        nnunet_thread = threading.Thread(target=self.runCommand,
                                         args=(nnunet_cmd, True, self.nnunet_process_finished_event,
                                               nnunet_progress_pattern, original_input_files_map))
        nnunet_thread.start()
        nnunet_thread.join()

        # --- Run roi2rdf ---
        logger.info("[INFO] Running roi2rdf on all files in nnunet output dir...")
        roi2rdf_cmd = f"cd {nnunet_output_dir} && {self.roi2rdf_path} *.v"
        self.runCommand(roi2rdf_cmd, stream_output=True)

        # --- Copy outputs to final output_dir ---
        logger.info("[INFO] Copying all .v and .rdf outputs to final destination...")

        copy_outputs_cmd = (
            f"cd {self.localTempDir.name} && "
            f"cp -- *.v {shlex.quote(output_dir)} && "
            f"cd {nnunet_output_dir} && "
            f"cp -- *.v {shlex.quote(output_dir)} && "
            f"cp -- *.rdf {shlex.quote(output_dir)}"
        )
        self.runCommand(copy_outputs_cmd, stream_output=True)

        # --- Rename the output files ---
        logger.info("[INFO] Renaming nnUNet output files...")
        try:
            for filename in os.listdir(output_dir):
                if filename.endswith('_0000.v'):
                    new_filename = filename.replace('_0000.v', '.v')
                    old_path = os.path.join(output_dir, filename)
                    new_path = os.path.join(output_dir, new_filename)
                    os.rename(old_path, new_path)
                    logger.debug(f"Renamed {old_path} to {new_path}")
        except Exception as e:
            logger.error(f"[ERROR] Error renaming files: {e}")

class MarAiRemote(MarAiBase):
    def __init__(self, hostname=None, username=None, password=None, ssh_keys=None, cfg_path=None, stop_event=None, gpu_id=None):
        super().__init__(cfg_path, stop_event=stop_event)

        # find matching host config
        selected_host_cfg = self._find_host_config(self.cfg.get('machines', []), hostname)
        if not selected_host_cfg:
            raise ValueError("[ERROR] No matching remote device found in config.\n")

        # extract details from config
        self.host_cfg = selected_host_cfg
        self.gpu_id = gpu_id
        self.hostname = selected_host_cfg.get('hostname')
        self.ipaddress = selected_host_cfg.get('ip')

        self.username = username
        self.password = password

        if not self.hostname:
            raise ValueError("[ERROR] Hostname not specified or found in configuration.\n")

        self.tempId = f"{os.getpid()}-{threading.get_ident()}"

        self.connected = False
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.sftp = None
        self.ssh_keys = ssh_keys

    def __del__(self):
        if hasattr(self, "connected") and self.connected:
            self.disconnect()

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

    def connect(self):
        if not self.connected:
            logger.info(f"[INFO] Connecting to remote host {self.hostname}...")
            try:
                if len(self.ssh_keys) > 0:
                    self.ssh.connect(self.hostname, port=22, username=self.username, key_filename=self.ssh_keys[0])
                    logger.info(f"[INFO] Connection successful via key-based authentication ({self.ssh_keys[0]}).")
                else:
                    self.ssh.connect(self.hostname, port=22, username=self.username, password=self.password)
                    logger.info(f"[INFO] Connection successful via password authentication")

                self.sftp = self.ssh.open_sftp()
                self.connected = True
            except Exception as e:
                logger.error(f"Connection failed: {e}")

    def disconnect(self):
        if self.connected:
            if self.sftp:
                try:
                    self.sftp.close()
                except Exception as e:
                    logger.warning(f"[ERROR] Error closing SFTP client: {e}")
            if self.ssh: # check if ssh object exists before closing
                try:
                    self.ssh.close()
                except Exception as e:
                    logger.warning(f"[ERROR] Error closing SSH client: {e}")
        self.connected = False
        self.sftp = None

        # reinit ssh client for next connection
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # run command remotely and stream output (now can signal completion)
    def runCommand(self, cmd, stream_output=False, process_event_on_completion=None,
                   progress_pattern=None, original_input_files_map=None):
        if not self.connected:
            raise RuntimeError("[ERROR] Not connected to remote host.")

        if isinstance(cmd, list):
            cmd_quoted = [shlex.quote(str(c)) for c in cmd]
            cmd_str = ' '.join(cmd_quoted)
        else:
            cmd_str = str(cmd)

        # add CUDA_VISIBLE_DEVICES for GPU-relevant commands
        gpu_tools = [
            os.path.basename(self.mic2ecat_path),
            os.path.basename(self.roi2rdf_path),
            "nnUNetv2_predict"
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

    def _check_file_existence(self, file_path):
        if not self.sftp:
            raise RuntimeError("[ERROR] SFTP client not connected.")
        try:
            self.sftp.stat(file_path)
            return True
        except FileNotFoundError:
            return False
        except Exception as e:
            logger.error(f"[ERROR] Error checking remote file existence for {file_path}: {e}")
            return False

    def _list_directory(self, directory_path):
        if not self.sftp:
            raise RuntimeError("[ERROR] SFTP client not connected.")
        return self.sftp.listdir(directory_path)

    def predictCall(self, input_files, microscope_number, output_dir, progress_callback=None):
        self.progress_callback = progress_callback
        self.total_expected_files = len(input_files)
        self.processed_files_count = 0
        self.reported_nnunet_files.clear()
        self.reported_rdf_files.clear()
        self.nnunet_process_finished_event.clear()

        if self.stop_event and self.stop_event.is_set():
            logger.info("[INFO] Remote prediction aborted before start.")
            return

        self.connect()
        remote_temp = f"/tmp/marai-{self.tempId}"
        nnunet_input = f"{remote_temp}/nnunet_input"
        nnunet_output = f"{remote_temp}/nnunet_output"
        self.runCommand(["mkdir", "-p", remote_temp, nnunet_input, nnunet_output])

        original_input_files_map = {self.get_file_prefix(f): f for f in input_files}

        # --- In-memory packaging of input files ---
        logger.info(f"[INFO] Packaging {len(input_files)} files for in-memory upload...")
        local_input_tar_fo = io.BytesIO()
        try:
            with tarfile.open(fileobj=local_input_tar_fo, mode='w:gz') as tar:
                for file_path in input_files:
                    tar.add(file_path, arcname=os.path.basename(file_path))
        except Exception as e:
            logger.error(f"[ERROR] Error creating in-memory tar archive: {e}")
            self.runCommand(f"rm -rf {remote_temp}")
            self.disconnect()
            raise

        # --- Upload the in-memory archive ---
        local_input_tar_fo.seek(0)
        remote_input_tar_path = f"{remote_temp}/input_{self.tempId}.tar.gz"
        logger.info(f"[INFO] Uploading input archive to {remote_temp}...")
        self.sftp.putfo(local_input_tar_fo, remote_input_tar_path)
        logger.info("[INFO] Input archive uploaded.")

        # --- Remote unpacking of input files ---
        logger.info(f"[INFO] Extracting input files on remote host...")
        self.runCommand(f"tar -xzf {remote_input_tar_path} -C {remote_temp}")

        # --- Run mic2ecat in batch remotely ---
        logger.info("[INFO] Running mic2ecat remotely on all files in temp dir...")
        mic2ecat_cmd = f"cd {remote_temp} && {self.mic2ecat_path} -j {microscope_number} *.tif"
        self.runCommand(mic2ecat_cmd, stream_output=True)

        # --- Copy .v outputs into nnunet_input dir remotely ---
        logger.info("[INFO] Moving and renaming mic2ecat outputs into nnunet_input directory...")

        remote_copy_cmd = (
            f"cd {remote_temp} && "
            f"for file in *.v; do "
            f"  prefix=$(basename $file .v); "
            f"  mv -- \"$file\" \"{nnunet_input}/${{prefix}}_0000.v\"; "
            f"done"
        )
        self.runCommand(remote_copy_cmd, stream_output=True)

        # --- Run nnUNet remotely ---
        nnunet_cmd = self.get_nnunet_base_cmd(nnunet_input, nnunet_output)
        device_flag = self._get_device_flag(self.host_cfg)
        nnunet_cmd_full = f"{nnunet_cmd} {device_flag}"
        nnunet_progress_pattern = r"done with (\S+)"
        nnunet_thread = threading.Thread(target=self.runCommand,
                                         args=(nnunet_cmd_full, True, self.nnunet_process_finished_event,
                                               nnunet_progress_pattern, original_input_files_map))
        nnunet_thread.start()
        nnunet_thread.join()

        # --- Run roi2rdf in batch remotely ---
        logger.info("[INFO] Running roi2rdf remotely on all files in nnunet output dir...")
        roi2rdf_cmd = f"cd {nnunet_output} && {self.roi2rdf_path} *.v"
        self.runCommand(roi2rdf_cmd, stream_output=True)

        # --- Remote packaging of output files ---
        logger.info("[INFO] Packaging output files for download...")

        remote_output_tar_path = f"{remote_temp}/output_{self.tempId}.tar.gz"

        remote_tar_cmd = (
            f"cd {remote_temp} && "
            f"tar -czf {remote_output_tar_path} "
            f"-C {nnunet_input} . "
            f"-C {nnunet_output} . "
        )

        self.runCommand(remote_tar_cmd, stream_output=True)

        if self.progress_callback:
            self.progress_callback(self.total_expected_files, self.total_expected_files, "Output files",
                                   "Packaging results")

        # --- Download the output archive into memory ---
        logger.info(f"[INFO] Downloading output archive...")
        local_output_tar_fo = io.BytesIO()
        self.sftp.getfo(remote_output_tar_path, local_output_tar_fo)
        logger.info("[INFO] Output archive downloaded.")

        # --- Unpack the in-memory output archive ---
        logger.info(f"[INFO] Unpacking output archive to {output_dir}...")
        local_output_tar_fo.seek(0)
        with tarfile.open(fileobj=local_output_tar_fo, mode='r:gz') as tar:
            tar.extractall(path=output_dir)

        # --- Rename the output files ---
        logger.info("[INFO] Renaming nnUNet output files...")
        try:
            for filename in os.listdir(output_dir):
                if filename.endswith('_0000.v'):
                    new_filename = filename.replace('_0000.v', '.v')
                    old_path = os.path.join(output_dir, filename)
                    new_path = os.path.join(output_dir, new_filename)
                    os.rename(old_path, new_path)
        except Exception as e:
            logger.error(f"[ERROR] Error renaming files: {e}")

        # --- Cleanup ---
        self.runCommand(f"rm -rf {remote_temp}")
        self.disconnect()