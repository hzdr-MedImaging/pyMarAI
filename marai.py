import os
import shutil
import subprocess
import tempfile
import platform
import yaml
import paramiko
import shlex
import threading


class MarAiBase:
    def __init__(self, cfg_path=None):
        # load config file
        cfg_path = cfg_path or ("C:\\pymarai\\pymarai.yml" if platform.system() == 'Windows' else "/home/melnyk80/PycharmProjects/PythonProject/pymarai.yml")
        self.cfg = self._load_config(cfg_path)

        # get paths for mic2ecat and roi2rdf
        try:
            self.mic2ecat_path = self.cfg['utils']['mic2ecat']
            self.roi2rdf_path = self.cfg['utils']['roi2Rdf']
        except KeyError as e:
            raise RuntimeError(f"Missing required path in config: {e}.\n")

        # get nnunet section from config
        self.nnunet_cfg = self.cfg.get('nnunet', {})
        self.localTempDir = tempfile.TemporaryDirectory(prefix="marai-")
        self.tempId = self.localTempDir.name.split('-')[1]

    # clean up temporary directory
    def __del__(self):
        try:
            self.localTempDir.cleanup()
        except Exception:
            pass

    # config file loader
    def _load_config(self, cfg_path):
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Config file not found: {cfg_path}.\n")
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
            raise ValueError("Missing one or more required nnUNet config fields.\n")

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

    def runCommand(self, cmd):
        raise NotImplementedError

    def predictFiles(self, input_files, microscope_number, output_dir):
        raise NotImplementedError

    def get_file_prefix(self, input_file):
        return os.path.splitext(os.path.basename(input_file))[0]


class MarAiLocal(MarAiBase):
    def __init__(self, cfg_path=None):
        super().__init__(cfg_path)

    # run a local shell command
    def runCommand(self, cmd):
        if isinstance(cmd, list):
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        else:
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, shell=True, executable="/bin/bash" if platform.system() != 'Windows' else None)
        output, _ = process.communicate()
        exit_status = process.wait()
        print(output)
        if exit_status != 0:
            raise RuntimeError(f"Command failed with exit status {exit_status}: {cmd}.\n")

    def predictCall(self, input_files, microscope_number, output_dir):
        # prepare input/output directories
        nnunet_input_dir = os.path.join(self.localTempDir.name, "nnunet_input")
        nnunet_output_dir = os.path.join(self.localTempDir.name, "nnunet_output")
        os.makedirs(nnunet_input_dir, exist_ok=True)
        os.makedirs(nnunet_output_dir, exist_ok=True)

        # convert each file and copy to nnunet input dir
        for input_file in input_files:
            file_prefix = self.get_file_prefix(input_file)
            temp_input_path = shutil.copy(input_file, self.localTempDir.name)
            v_path = os.path.join(self.localTempDir.name, f"{file_prefix}.v")

            self.runCommand([self.mic2ecat_path, "-j", str(microscope_number), temp_input_path])
            shutil.copy(v_path, os.path.join(nnunet_input_dir, f"{file_prefix}_0000.v"))

        # run nnunet
        nnunet_cmd = self.get_nnunet_base_cmd(nnunet_input_dir, nnunet_output_dir)
        nnunet_cmd += f" -device cpu"
        self.runCommand(nnunet_cmd)

        # convert and copy output files
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

        # find matching host config
        selected_host_cfg = self._find_host_config(self.cfg.get('machines', []), hostname)
        if not selected_host_cfg:
            raise ValueError("No matching remote device found in config.\n")

        # extract details from config
        self.host_cfg = selected_host_cfg
        self.hostname = selected_host_cfg.get('hostname')
        self.ipaddress = selected_host_cfg.get('ip')

        self.username = username
        self.password = password

        if not self.hostname:
            raise ValueError("Hostname not specified or found in configuration.\n")

        self.tempId = os.getpid()
        self.connected = False
        self.ssh = paramiko.SSHClient()
        self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    def __del__(self):
        if hasattr(self, "connected") and self.connected:
            self.disconnect()

    # find device config by hostname or use default
    def _find_host_config(self, remote_list, hostname=None):
        for entry in remote_list:
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
            self.ssh.connect(self.hostname, username=self.username, password=self.password)
            self.connected = True

    def disconnect(self):
        if self.connected:
            self.ssh.close()
        self.connected = False

    # run command remotely and stream output
    def runCommand(self, cmd):
        if isinstance(cmd, list):
            cmd = ['stdbuf', '-o0'] + [shlex.quote(str(c)) for c in cmd]
            cmd_str = ' '.join(cmd)
        else:
            cmd_str = f"stdbuf -o0 {cmd}"

        stdin, stdout, stderr = self.ssh.exec_command(cmd_str, get_pty=True)

        def reader(stream):
            for line in iter(stream.readline, ""):
                print(line, end='')

        threads = [threading.Thread(target=reader, args=(stream,)) for stream in (stdout, stderr)]
        for t in threads: t.start()
        for t in threads: t.join()

        exit_status = stdout.channel.recv_exit_status()
        if exit_status != 0:
            raise RuntimeError(f"Command failed with exit status {exit_status}: {cmd_str}.\n")

    def predictCall(self, input_files, microscope_number, output_dir):
        try:
            # connect and prepare remote temp dirs
            self.connect()
            remote_temp = f"/tmp/marai-{self.tempId}"
            nnunet_input = f"{remote_temp}/nnunet_input"
            nnunet_output = f"{remote_temp}/nnunet_output"
            self.runCommand(["mkdir", "-p", remote_temp, nnunet_input, nnunet_output])

            # upload files and preprocess
            with self.ssh.open_sftp() as sftp:
                for input_file in input_files:
                    file_prefix = self.get_file_prefix(input_file)
                    ext = os.path.splitext(input_file)[1]
                    remote_input = f"{remote_temp}/{file_prefix}{ext}"
                    sftp.put(input_file, remote_input)
                    v_path = f"{remote_temp}/{file_prefix}.v"
                    self.runCommand([self.mic2ecat_path, "-j", str(microscope_number), remote_input])
                    self.runCommand(["cp", v_path, f"{nnunet_input}/{file_prefix}_0000.v"])

            # run nnunet remotely
            nnunet_cmd = self.get_nnunet_base_cmd(nnunet_input, nnunet_output)
            device_flag = self._get_device_flag(self.host_cfg)
            nnunet_cmd_full = f"{nnunet_cmd} {device_flag}"
            self.runCommand(nnunet_cmd_full)

            # download results
            with self.ssh.open_sftp() as sftp:
                os.makedirs(output_dir, exist_ok=True)
                for input_file in input_files:
                    file_prefix = self.get_file_prefix(input_file)
                    pred_v = f"{nnunet_output}/{file_prefix}.v"
                    self.runCommand([self.roi2rdf_path, pred_v])
                    rdf_path = f"{nnunet_output}/{file_prefix}.rdf"
                    local_v_path = os.path.join(output_dir, f"{file_prefix}.v")
                    local_rdf_path = os.path.join(output_dir, f"{file_prefix}.rdf")
                    sftp.get(pred_v, local_v_path)
                    sftp.get(rdf_path, local_rdf_path)

            # cleanup remote temp
            self.runCommand(["rm", "-rf", remote_temp])

        except Exception as e:
            raise