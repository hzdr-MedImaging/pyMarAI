import os
import shutil
import subprocess
import tempfile
import platform
import yaml
import re
import paramiko

class MarAiBase:
    def __init__(self, mic2ecat_path, roi2rdf_path):
        self.mic2ecat_path = mic2ecat_path
        self.roi2rdf_path = roi2rdf_path
        self.localTempDir = tempfile.TemporaryDirectory(prefix="marai-")
        self.tempId = self.localTempDir.name.split('-')[1]

    def __del__(self):
        try:
            self.localTempDir.cleanup()
        except Exception:
            pass

    def _runCommand(self, cmd, shell=False):
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True, shell=shell, executable="/bin/bash" if shell else None
        )
        output, _ = process.communicate()
        exit_status = process.wait()
        print(output)
        if exit_status != 0:
            raise RuntimeError(f"Command failed with exit status {exit_status}: {cmd}")

    def _prepareNnUNetInput(self, v_path, input_dir, file_prefix):
        os.makedirs(input_dir, exist_ok=True)
        dest_file = os.path.join(input_dir, f"{file_prefix}_0000.v")
        shutil.copy(v_path, dest_file)

    def _copyOutputFiles(self, v_path, rdf_path, output_dir, file_prefix):
        os.makedirs(output_dir, exist_ok=True)
        shutil.copy(v_path, os.path.join(output_dir, f"{file_prefix}.v"))
        shutil.copy(rdf_path, os.path.join(output_dir, f"{file_prefix}.rdf"))


class MarAiLocal(MarAiBase):
    def __init__(self, mic2ecat_path="/usr/local/petlib/bin/mic2ecat",
                 roi2rdf_path="/usr/local/petlib/bin/roi2Rdf"):
        super().__init__(mic2ecat_path, roi2rdf_path)

    def predictSingleFile(self, input_file, microscope_number, output_dir):
        file_prefix = os.path.splitext(os.path.basename(input_file))[0]
        temp_input_path = shutil.copy(input_file, self.localTempDir.name)

        v_path = os.path.join(self.localTempDir.name, f"{file_prefix}.v")
        self._runCommand([self.mic2ecat_path, "-j", str(microscope_number), temp_input_path])

        nnunet_input_dir = os.path.join(self.localTempDir.name, "nnunet_input")
        nnunet_output_dir = os.path.join(self.localTempDir.name, "nnunet_output")
        self._prepareNnUNetInput(v_path, nnunet_input_dir, file_prefix)
        os.makedirs(nnunet_output_dir, exist_ok=True)

        cmd = (
            f"conda run -n nnunet-spheroids --live-stream nnUNetv2_predict "
            f"-d Dataset001_spheroids_V1 -i {nnunet_input_dir} -o {nnunet_output_dir} "
            f"-f 0 1 2 3 4 -tr nnUNetTrainer_noSmooth -c 2d -p nnUNetPlans -device cpu"
        )
        self._runCommand(cmd, shell=True)

        pred_v = os.path.join(nnunet_output_dir, f"{file_prefix}.v")
        self._runCommand([self.roi2rdf_path, pred_v])
        rdf_path = os.path.splitext(pred_v)[0] + ".rdf"

        self._copyOutputFiles(pred_v, rdf_path, output_dir, file_prefix)


class MarAiRemote(MarAiBase):
    def __init__(self, hostname=None, ipaddress=None, username=None, password=None,
                 mic2ecat_path="/usr/local/petlib/bin/mic2ecat",
                 roi2rdf_path="/usr/local/petlib/bin/roi2Rdf"):
        super().__init__(mic2ecat_path, roi2rdf_path)

        cfg_path = "C:\\pymarai\\pymarai.yml" if platform.system() == 'Windows' else "/usr/local/etc/pymarai.yml"
        cfg = self._load_config(cfg_path, hostname)
        self.hostname = cfg.get('hostname', hostname)
        self.ipaddress = cfg.get('ip', ipaddress)
        self.username = cfg.get('username', username)
        self.password = cfg.get('password', password)

        self.connected = False
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    def __del__(self):
        self.disconnect()
        super().__del__()

    def _load_config(self, cfg_file, hostname):
        if not os.path.exists(cfg_file):
            return {}
        with open(cfg_file, 'r') as f:
            cfg = yaml.safe_load(f)
        for marai in cfg.get('marai', []):
            host = list(marai.keys())[0]
            if hostname is None or re.match("^.*" + hostname, host):
                return {
                    'hostname': host,
                    'ip': marai[host][0],
                    'username': marai[host][1],
                    'password': marai[host][2],
                    'marai_utils_path': cfg.get('marai-util')
                }
        return {}

    def connect(self):
        if not self.connected:
            self.ssh.connect(self.ipaddress, username=self.username, password=self.password)
            self.connected = True

    def disconnect(self):
        if self.connected:
            self.ssh.close()
        self.connected = False

    def _execRemote(self, command):
        stdin, stdout, stderr = self.ssh.exec_command(command)
        out = stdout.read().decode()
        err = stderr.read().decode()
        print(out)
        if err.strip():
            raise RuntimeError(f"Remote error: {err.strip()}")

    def predictSingleFile(self, input_file, microscope_number, output_dir):
        self.connect()
        file_prefix = os.path.splitext(os.path.basename(input_file))[0]
        ext = os.path.splitext(input_file)[1]

        remote_temp = f"/tmp/marai-{self.tempId}"
        remote_input = f"{remote_temp}/{file_prefix}{ext}"
        remote_output = f"{remote_temp}/output"

        self._execRemote(f"mkdir -p {remote_temp} {remote_output}")

        with self.ssh.open_sftp() as sftp:
            sftp.put(input_file, remote_input)

        v_path = f"{remote_temp}/{file_prefix}.v"
        self._execRemote(f"{self.mic2ecat_path} -j {microscope_number} {remote_input}")

        nnunet_input = f"{remote_temp}/nnunet_input"
        self._execRemote(f"mkdir -p {nnunet_input} && cp {v_path} {nnunet_input}/{file_prefix}_0000.v")

        nnunet_cmd = (
            f"conda run -n --live-stream nnUNetv2_predict -d Dataset001_spheroids_V1 "
            f"-i {nnunet_input} -o {remote_output} -f 0 1 2 3 4 -tr nnUNetTrainer_noSmooth -c 2d -p nnUNetPlans"
        )
        self._execRemote(nnunet_cmd)

        pred_v = f"{remote_output}/{file_prefix}.v"
        self._execRemote(f"{self.roi2rdf_path} {pred_v}")
        rdf_path = f"{remote_output}/{file_prefix}.rdf"

        with self.ssh.open_sftp() as sftp:
            os.makedirs(output_dir, exist_ok=True)
            sftp.get(pred_v, os.path.join(output_dir, f"{file_prefix}.v"))
            sftp.get(rdf_path, os.path.join(output_dir, f"{file_prefix}.rdf"))

        self._execRemote(f"rm -rf {remote_temp}")