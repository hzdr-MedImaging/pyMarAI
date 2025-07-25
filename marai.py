import os
import shutil
import subprocess
import tempfile
import platform
import yaml
import paramiko
import threading
import time
import logging
import shlex
import re

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
        cfg_path = cfg_path or (
            "C:\\pymarai\\pymarai.yml" if platform.system() == 'Windows' else "/home/melnyk80/PycharmProjects/PythonProject/pymarai.yml")
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
            f"conda run -n {env} --live-stream nnUNetv2_predict "
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
        self.nnunet_process_finished_event.clear()  # Reset for each call

        if self.stop_event and self.stop_event.is_set():
            logger.info("[INFO] Local prediction aborted by external signal before start.")
            return

        nnunet_input_dir = os.path.join(self.localTempDir.name, "nnunet_input")
        nnunet_output_dir = os.path.join(self.localTempDir.name, "nnunet_output")
        os.makedirs(nnunet_input_dir, exist_ok=True)
        os.makedirs(nnunet_output_dir, exist_ok=True)

        original_input_files_map = {self.get_file_prefix(f): f for f in input_files}

        logger.info(f"[INFO] Preparing {self.total_expected_files} files for nnUNet...")
        # Add a counter for the "Preparing files" stage
        prepared_files_count = 0
        for input_file in input_files:
            if self.stop_event and self.stop_event.is_set():
                logger.info("[INFO] Local prediction aborted by external signal during preprocessing.")
                return

            file_prefix = self.get_file_prefix(input_file)
            temp_input_path_for_mic2ecat = os.path.join(self.localTempDir.name, os.path.basename(input_file))
            shutil.copy(input_file, temp_input_path_for_mic2ecat)

            v_path = os.path.join(self.localTempDir.name, f"{file_prefix}.v")
            self.runCommand([self.mic2ecat_path, "-j", str(microscope_number), temp_input_path_for_mic2ecat])
            shutil.copy(v_path, os.path.join(nnunet_input_dir, self._get_nnunet_input_name_for_temp_copy(file_prefix)))

            # Report progress for "Preparing files for prediction"
            prepared_files_count += 1
            if self.progress_callback:
                self.progress_callback(
                    prepared_files_count,
                    self.total_expected_files,
                    os.path.basename(input_file),
                    "Preparing files for prediction"  # New stage indicator
                )

        logger.info("[INFO] Finished uploading and preprocessing.")

        nnunet_cmd = self.get_nnunet_base_cmd(nnunet_input_dir, nnunet_output_dir)
        nnunet_cmd += f" -device cpu"
        logger.info(f"[INFO] Running nnUNet command: {nnunet_cmd}")

        # The progress pattern for nnUNet already captures the filename.
        # The stage indicator for nnUNet is already set in runCommand.
        nnunet_progress_pattern = r"done with (\S+)"

        # nnUNet execution now directly reports progress via runCommand's streaming
        nnunet_execution_thread = threading.Thread(target=self.runCommand,
                                                   args=(nnunet_cmd, True, self.nnunet_process_finished_event,
                                                         nnunet_progress_pattern, original_input_files_map))
        nnunet_execution_thread.daemon = True
        nnunet_execution_thread.start()

        # wait for the nnUNet execution thread to complete
        try:
            nnunet_execution_thread.join()
        except Exception as e:
            logger.error(f"[ERROR] An error occurred in the nnUNet execution thread: {e}")
            raise

        logger.info("[INFO] nnUNet prediction command completed (execution thread finished).")

        # post-processing (roi2rdf and copying to final output_dir)
        rdf_processed_count = 0
        for input_file in input_files:
            if self.stop_event and self.stop_event.is_set():
                logger.info("[INFO] Local prediction aborted by external signal during post-processing.")
                break

            file_prefix = self.get_file_prefix(input_file)
            nnunet_v_path_in_temp, rdf_path_in_temp = self._get_expected_full_paths_in_nnunet_output_dir(
                nnunet_output_dir, file_prefix)

            # ensure roi2rdf is run
            if self._check_file_existence(nnunet_v_path_in_temp):
                self.runCommand([self.roi2rdf_path, nnunet_v_path_in_temp], stream_output=True)
                try:
                    self._wait_for_file_existence(rdf_path_in_temp)
                except TimeoutError as e:
                    logger.warning(f"Warning: {e}. RDF for {file_prefix} might be missing or delayed.")

            local_v_path = os.path.join(output_dir, f"{file_prefix}.v")
            local_rdf_path = os.path.join(output_dir, self._get_roi2rdf_output_name_in_temp(file_prefix))

            mic2ecat_v_path = os.path.join(self.localTempDir.name, f"{file_prefix}.v")
            if self._check_file_existence(mic2ecat_v_path) and not os.path.exists(local_v_path):
                shutil.copy(mic2ecat_v_path, local_v_path)

            if self._check_file_existence(rdf_path_in_temp) and not os.path.exists(local_rdf_path):
                shutil.copy(rdf_path_in_temp, local_rdf_path)

            if os.path.exists(local_rdf_path) and file_prefix not in self.reported_rdf_files:
                rdf_processed_count += 1
                self.reported_rdf_files.add(file_prefix)
                if self.progress_callback:
                    self.progress_callback(
                        rdf_processed_count,
                        self.total_expected_files,
                        os.path.basename(input_file),
                        "Making ROI mask files"  # Changed label here to be more descriptive
                    )
                logger.info(
                    f"Reported RDF progress for {file_prefix}: {rdf_processed_count}/{self.total_expected_files}")

        logger.info("[INFO] All files post-processed and progress confirmed.")


class MarAiRemote(MarAiBase):
    def __init__(self, hostname=None, ipaddress=None, username=None, password=None, cfg_path=None, stop_event=None):
        super().__init__(cfg_path, stop_event=stop_event)

        # find matching host config
        selected_host_cfg = self._find_host_config(self.cfg.get('machines', []), hostname)
        if not selected_host_cfg:
            raise ValueError("[ERROR] No matching remote device found in config.\n")

        # extract details from config
        self.host_cfg = selected_host_cfg
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
            self.ssh.connect(self.hostname, username=self.username, password=self.password)
            self.sftp = self.ssh.open_sftp()
            self.connected = True
            logger.info("[INFO] Successfully connected.")

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
        self.ssh = paramiko.SSHClient() # reinitialize for next connection
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    # run command remotely and stream output (now can signal completion)
    def runCommand(self, cmd, stream_output=False, process_event_on_completion=None,
                   progress_pattern=None, original_input_files_map=None):
        if not self.connected:
            raise RuntimeError("[ERROR] Not connected to remote host.")

        if isinstance(cmd, list):
            cmd_quoted = [shlex.quote(str(c)) for c in cmd]
            cmd_str = 'stdbuf -o0 ' + ' '.join(cmd_quoted)
        else:
            cmd_str = f"stdbuf -o0 {cmd}"

        stdin, stdout, stderr = self.ssh.exec_command(cmd_str, get_pty=True)

        if stream_output:
            for line in iter(stdout.readline, ""):
                print(line, end='')
                if progress_pattern and self.progress_callback and original_input_files_map:
                    match = re.search(progress_pattern, line)
                    if match:
                        filename = match.group(1) # Assuming group 1 captures the filename/prefix
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
            # Also process stderr for potential progress messages, though less common for "Done"
            for line in iter(stderr.readline, ""):
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
                                "nnunet_predicting"
                            )
                            logger.info(f"Reported nnUNet progress for {file_prefix_from_output} via stderr: {current_nnunet_count}/{self.total_expected_files}")
        else:
            output = stdout.read().decode('utf-8') + stderr.read().decode('utf-8')
            print(output)

        exit_status = stdout.channel.recv_exit_status()
        if exit_status != 0:
            raise RuntimeError(f"[ERROR] Remote command failed with exit status {exit_status}: {cmd_str}.\n")

        if process_event_on_completion:
            process_event_on_completion.set()  # signal that this command has completed

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

    def predictCall(self, input_files, microscope_number, output_dir,
                    progress_callback=None):
        self.progress_callback = progress_callback
        self.total_expected_files = len(input_files)
        self.processed_files_count = 0
        self.reported_nnunet_files.clear()
        self.reported_rdf_files.clear()
        self.nnunet_process_finished_event.clear()
        remote_temp = None

        if self.stop_event and self.stop_event.is_set():
            logger.info("[INFO] Remote prediction aborted by external signal before start.")
            return

        try:
            self.connect()
            remote_temp = f"/tmp/marai-{self.tempId}"
            nnunet_input = f"{remote_temp}/nnunet_input"
            nnunet_output = f"{remote_temp}/nnunet_output"
            self.runCommand(["mkdir", "-p", remote_temp, nnunet_input, nnunet_output], stream_output=True)

            original_input_files_map = {self.get_file_prefix(f): f for f in input_files}

            logger.info(f"[INFO] Preparing {self.total_expected_files} files for nnUNet...")
            # Add a counter for the "Preparing files" stage
            prepared_files_count = 0
            for i, input_file in enumerate(input_files):
                if self.stop_event and self.stop_event.is_set():
                    logger.info("[INFO] Remote prediction aborted by external signal during upload.")
                    raise UserCancelledError("Remote upload cancelled by user.")

                file_prefix = self.get_file_prefix(input_file)
                ext = os.path.splitext(input_file)[1]
                remote_input = f"{remote_temp}/{file_prefix}{ext}"
                self.sftp.put(input_file, remote_input)

                v_path = f"{remote_temp}/{file_prefix}.v"
                self.runCommand([self.mic2ecat_path, "-j", str(microscope_number), remote_input], stream_output=True)
                self.runCommand(
                    ["cp", v_path, f"{nnunet_input}/{self._get_nnunet_input_name_for_temp_copy(file_prefix)}"],
                    stream_output=True)

                # Report progress for "Preparing files for prediction"
                prepared_files_count += 1
                if self.progress_callback:
                    self.progress_callback(
                        prepared_files_count,
                        self.total_expected_files,
                        os.path.basename(input_file),
                        "Preparing files for prediction"  # New stage indicator
                    )

            logger.info("[INFO] Finished uploading and preprocessing.")

            nnunet_cmd = self.get_nnunet_base_cmd(nnunet_input, nnunet_output)
            device_flag = self._get_device_flag(self.host_cfg)
            nnunet_cmd_full = f"{nnunet_cmd} {device_flag}"
            logger.info(f"[INFO] Running nnUNet command: {nnunet_cmd_full}")

            nnunet_progress_pattern = r"done with (\S+)"

            # nnUNet execution now directly reports progress via runCommand's streaming
            nnunet_execution_thread = threading.Thread(target=self.runCommand,
                                                       args=(nnunet_cmd_full, True, self.nnunet_process_finished_event,
                                                             nnunet_progress_pattern, original_input_files_map))
            nnunet_execution_thread.daemon = True
            nnunet_execution_thread.start()

            # wait for the nnUNet execution thread to complete
            try:
                nnunet_execution_thread.join()
            except Exception as e:
                logger.error(f"[ERROR] An error occurred in the remote nnUNet execution thread: {e}")
                raise

            logger.info("[INFO] nnUNet prediction command completed (execution thread finished).")

            # post-processing (roi2rdf and copying to final output_dir)
            rdf_processed_count = 0
            for input_file in input_files:
                if self.stop_event and self.stop_event.is_set():
                    logger.info("[INFO] Remote prediction aborted by external signal during download/post-processing.")
                    break

                file_prefix = self.get_file_prefix(input_file)
                remote_nnunet_v_path_in_temp, remote_rdf_path_in_temp = self._get_expected_full_paths_in_nnunet_output_dir(
                    nnunet_output, file_prefix)

                if self._check_file_existence(remote_nnunet_v_path_in_temp):
                    self.runCommand([self.roi2rdf_path, remote_nnunet_v_path_in_temp], stream_output=True)
                    try:
                        self._wait_for_file_existence(remote_rdf_path_in_temp)
                    except TimeoutError as e:
                        logger.warning(f"Warning: {e}. RDF for {file_prefix} might be missing or delayed on remote.")

                local_v_path = os.path.join(output_dir, f"{file_prefix}.v")
                local_rdf_path = os.path.join(output_dir, self._get_roi2rdf_output_name_in_temp(file_prefix))

                remote_v_path = f"{remote_temp}/{file_prefix}.v"  # This is the mic2ecat output .v file, not nnunet
                if self._check_file_existence(remote_v_path) and not os.path.exists(local_v_path):
                    self.sftp.get(remote_v_path, local_v_path)

                if self._check_file_existence(remote_rdf_path_in_temp) and not os.path.exists(local_rdf_path):
                    self.sftp.get(remote_rdf_path_in_temp, local_rdf_path)

                if os.path.exists(local_rdf_path) and file_prefix not in self.reported_rdf_files:
                    rdf_processed_count += 1
                    self.reported_rdf_files.add(file_prefix)
                    if self.progress_callback:
                        self.progress_callback(
                            rdf_processed_count,
                            self.total_expected_files,
                            os.path.basename(input_file),
                            "Making ROI mask files"
                        )
                    logger.info(
                        f"Reported RDF progress for {file_prefix}: {rdf_processed_count}/{self.total_expected_files}")

            logger.info("[INFO] All files post-processed, downloaded, and progress confirmed.")

            logger.info(f"[INFO] Cleaning up remote temporary directory: {remote_temp}")
            self.runCommand(["rm", "-rf", remote_temp], stream_output=True)
            logger.info("[INFO] Remote cleanup complete.")

        except UserCancelledError:
            logger.info("[INFO] Remote prediction cancelled by user.")
            raise

        except Exception as e:
            logger.error(f"[ERROR] An error occurred in MarAiRemote: {e}")
            raise

        finally:
            self.disconnect()
            logger.info("[INFO] Disconnected from remote host.")
            if remote_temp and self.connected:  # Check self.connected as disconnect might fail
                try:
                    self.runCommand(["rm", "-rf", remote_temp], stream_output=True)
                    logger.info("[INFO] Remote cleanup complete in finally block.")
                except Exception as cleanup_e:
                    logger.warning(f"[WARNING] Failed to clean up remote temp dir {remote_temp}: {cleanup_e}")
