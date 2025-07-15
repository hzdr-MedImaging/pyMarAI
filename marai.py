import os
import shutil
import subprocess
import tempfile
import platform
import yaml
import paramiko
import shlex
import getpass

from login import LoginDialog
from PyQt5.QtWidgets import QDialog

class MarAiBase:
    def __init__(self, cfg_path=None):
        cfg_path = cfg_path or ("C:\\pymarai\\pymarai.yml" if platform.system() == 'Windows' else "/home/melnyk80/PycharmProjects/PythonProject/pymarai.yml")
        self.cfg = self._load_config(cfg_path)

        try:
            self.mic2ecat_path = self.cfg['utils']['mic2ecat']
            self.roi2rdf_path = self.cfg['utils']['roi2Rdf']
        except KeyError as e:
            raise RuntimeError(f"Missing required path in config: {e}")

        self.nnunet_cfg = self.cfg.get('nnunet', {})
        self.localTempDir = tempfile.TemporaryDirectory(prefix="marai-")
        self.tempId = self.localTempDir.name.split('-')[1]

    def __del__(self):
        try:
            self.localTempDir.cleanup()
        except Exception:
            pass

    def _load_config(self, cfg_path):
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        with open(cfg_path, 'r') as f:
            return yaml.safe_load(f)

    def get_nnunet_base_cmd(self, input_dir, output_dir):
        env = self.nnunet_cfg.get('env', 'nnunet')
        dataset = self.nnunet_cfg.get('dataset')
        trainer = self.nnunet_cfg.get('trainer')
        config = self.nnunet_cfg.get('config')
        plans = self.nnunet_cfg.get('plans')
        folds = self.nnunet_cfg.get('folds', [0])

        if not all([dataset, trainer, config, plans]):
            raise ValueError("Missing one or more required nnUNet config fields.")

        folds_str = ' '.join(map(str, folds))

        return (
            f"conda run -n {env} --live-stream nnUNetv2_predict "
            f"-d {dataset} -i {input_dir} -o {output_dir} "
            f"-f {folds_str} -tr {trainer} -c {config} -p {plans}"
        )

    def _get_device_flag(self, machine_name):
        for entry in self.cfg.get("machines", []):
            if machine_name in entry:
                machine = entry[machine_name]
                if isinstance(machine, dict):
                    if machine.get("cpu", False) and not machine.get("gpu", False):
                        return "-device cpu"
        return ""

    def runCommand(self, cmd):
        raise NotImplementedError

    def predictFiles(self, input_files, microscope_number, output_dir):
        raise NotImplementedError

    def get_file_prefix(self, input_file):
        return os.path.splitext(os.path.basename(input_file))[0]


class MarAiLocal(MarAiBase):
    def __init__(self, cfg_path=None):
        super().__init__(cfg_path)

    def runCommand(self, cmd):
        if isinstance(cmd, list):
            cmd_str = ' '.join(shlex.quote(str(c)) for c in cmd)
            shell = False
        else:
            cmd_str = cmd
            shell = True

        process = subprocess.Popen(
            cmd_str, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True, shell=shell,
            executable="/bin/bash" if shell else None
        )
        output, _ = process.communicate()
        exit_status = process.wait()
        print(output)
        if exit_status != 0:
            raise RuntimeError(f"Command failed with exit status {exit_status}: {cmd}")

    def predictCall(self, input_files, microscope_number, output_dir):
        nnunet_input_dir = os.path.join(self.localTempDir.name, "nnunet_input")
        nnunet_output_dir = os.path.join(self.localTempDir.name, "nnunet_output")
        os.makedirs(nnunet_input_dir, exist_ok=True)
        os.makedirs(nnunet_output_dir, exist_ok=True)

        for input_file in input_files:
            file_prefix = self.get_file_prefix(input_file)
            temp_input_path = shutil.copy(input_file, self.localTempDir.name)
            v_path = os.path.join(self.localTempDir.name, f"{file_prefix}.v")

            self.runCommand([self.mic2ecat_path, "-j", str(microscope_number), temp_input_path])
            shutil.copy(v_path, os.path.join(nnunet_input_dir, f"{file_prefix}_0000.v"))

        nnunet_cmd = self.get_nnunet_base_cmd(nnunet_input_dir, nnunet_output_dir)
        nnunet_cmd += f" -device cpu"
        self.runCommand(nnunet_cmd)

        os.makedirs(output_dir, exist_ok=True)
        for input_file in input_files:
            file_prefix = self.get_file_prefix(input_file)
            pred_v = os.path.join(nnunet_output_dir, f"{file_prefix}.v")
            self.runCommand([self.roi2rdf_path, pred_v])
            rdf_path = os.path.splitext(pred_v)[0] + ".rdf"
            shutil.copy(pred_v, os.path.join(output_dir, f"{file_prefix}.v"))
            shutil.copy(rdf_path, os.path.join(output_dir, f"{file_prefix}.rdf"))


class MarAiRemote(MarAiBase):
    def __init__(self, hostname=None, ipaddress=None, username=None, password=None, cfg_path=None):
        super().__init__(cfg_path)

        cfg_remote = self._load_remote_config(self.cfg.get('machines', []), hostname)
        self.hostname = cfg_remote.get('hostname', hostname)
        self.ipaddress = cfg_remote.get('ip', ipaddress)

        # prompt if not provided
        self.username = username or cfg_remote.get('username') or input("SSH username: ")
        self.password = password or cfg_remote.get('password') or getpass.getpass("SSH password: ")

        self.connected = False
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    def __del__(self):
        if hasattr(self, "connected") and self.connected:
            self.disconnect()

    def _load_remote_config(self, remote_list, parent_widget=None):
        for marai in remote_list:
            host = list(marai.keys())[0]
            values = marai[host]
            ip = values[0]
            extras = values[1] if len(values) > 1 and isinstance(values[1], dict) else {}

            if extras.get("default", False):
                login_dialog = LoginDialog(parent=parent_widget)
                if login_dialog.exec_() == QDialog.Accepted:
                    username, password = login_dialog.get_credentials()
                else:
                    raise ValueError("SSH login canceled by user.")

                return {
                    'hostname': host,
                    'ip': ip,
                    'username': username,
                    'password': password
                }

        raise ValueError("No default remote device found in config.")

    def connect(self):
        if not self.connected:
            self.ssh.connect(self.ipaddress, username=self.username, password=self.password)
            self.connected = True

    def disconnect(self):
        if self.connected:
            self.ssh.close()
        self.connected = False

    def runCommand(self, cmd):
        if isinstance(cmd, list):
            cmd = ' '.join(shlex.quote(str(c)) for c in cmd)

        stdin, stdout, stderr = self.ssh.exec_command(cmd)
        exit_status = stdout.channel.recv_exit_status()
        out = stdout.read().decode()
        err = stderr.read().decode()
        print(out)
        if exit_status != 0:
            raise RuntimeError(f"Command failed: {cmd}\n{err.strip()}")

    def predictCall(self, input_files, microscope_number, output_dir):
        self.connect()
        remote_temp = f"/tmp/marai-{self.tempId}"
        nnunet_input = f"{remote_temp}/nnunet_input"
        nnunet_output = f"{remote_temp}/nnunet_output"

        self.runCommand(["mkdir", "-p", remote_temp, nnunet_input, nnunet_output])

        with self.ssh.open_sftp() as sftp:
            for input_file in input_files:
                file_prefix = self.get_file_prefix(input_file)
                ext = os.path.splitext(input_file)[1]
                remote_input = f"{remote_temp}/{file_prefix}{ext}"
                sftp.put(input_file, remote_input)

                v_path = f"{remote_temp}/{file_prefix}.v"
                self.runCommand([self.mic2ecat_path, "-j", str(microscope_number), remote_input])
                self.runCommand(["cp", v_path, f"{nnunet_input}/{file_prefix}_0000.v"])

        nnunet_cmd = self.get_nnunet_base_cmd(nnunet_input, nnunet_output)
        nnunet_cmd += f" {self._get_device_flag(self.hostname)}"
        self.runCommand(nnunet_cmd)

        with self.ssh.open_sftp() as sftp:
            os.makedirs(output_dir, exist_ok=True)
            for input_file in input_files:
                file_prefix = self.get_file_prefix(input_file)
                pred_v = f"{nnunet_output}/{file_prefix}.v"
                self.runCommand([self.roi2rdf_path, pred_v])
                rdf_path = f"{nnunet_output}/{file_prefix}.rdf"
                sftp.get(pred_v, os.path.join(output_dir, f"{file_prefix}.v"))
                sftp.get(rdf_path, os.path.join(output_dir, f"{file_prefix}.rdf"))

        self.runCommand(["rm", "-rf", remote_temp])