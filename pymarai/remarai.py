import json
import os
import shlex
import subprocess
import threading
import logging
from sys import platform
from typing import List, Tuple, Optional

from marai import MarAiRemote
from pymarai.config import AppConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class MarAiRemoteRetrain(MarAiRemote):
    def __init__(self, hostname=None, username=None, password=None, ssh_keys=None, stop_event=None):
        super().__init__(hostname=hostname, username=username, password=password,
                         ssh_keys=ssh_keys, stop_event=stop_event, gpu_id=None)

        # load retrain section from cfg
        # get config sections
        config = AppConfig()
        self.utils = config.get_utils()
        self.scripts = config.get_scripts()
        self.nnunet = config.get_nnunet()
        self.retrain = config.get_retrain()

        self.thrass_bin = self.utils["thrass"]
        self.conda_bin = self.utils["conda"]
        self.nnunet_train_bin = self.utils["nnunet_train"]

        self.nnunet_env  = self.nnunet["env"]
        self.config = self.nnunet["config"]
        self.nnunet_trainer = self.nnunet["trainer"]

        self.preprocess_script = self.scripts["preprocess_script"]
        self.create_dataset_script = self.scripts["create_dataset_script"]
        self.custom_split_script  = self.scripts["custom_split_script"]

        self.training_staging_dir = self.retrain["training_staging_dir"]
        self.nnunet_preprocessed_dir  = self.retrain["nnunet_preprocessed_dir"]
        self.dataset_workdir  = self.retrain["dataset_workdir"]

        required = {
            "thrass": self.thrass_bin,
            "conda": self.conda_bin,
            "nnunet_train": self.nnunet_train_bin,
            "nnunet_env": self.nnunet_env,
            "nnunet_config": self.config,
            "nnunet_trainer": self.nnunet_trainer,
            "preprocess_script": self.preprocess_script,
            "create_dataset_script": self.create_dataset_script,
            "custom_split_script": self.custom_split_script,
            "training_staging_dir": self.training_staging_dir,
            "nnunet_preprocessed_dir": self.nnunet_preprocessed_dir,
            "dataset_workdir": self.dataset_workdir,
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise ValueError(f"[ERROR] Missing retrain config keys: {', '.join(missing)}")

    def _exec_local(self, cmd: str, stream_output: bool = False) -> int:

        if isinstance(cmd, list):
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                env=os.environ.copy()
            )
        else:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                env=os.environ.copy(),
                shell=True,
                executable="/bin/bash" if platform.system() != "Windows" else None
            )

        if stream_output:
            for line in process.stdout:
                print(line, end="")
        else:
            output, _ = process.communicate()
            print(output)

        exit_status = process.wait()
        if exit_status != 0:
            raise RuntimeError(f"[ERROR] Command failed with exit status {exit_status}: {cmd}")

        return exit_status

    # run a remote command via SSH and optionally stream output
    def _exec_remote(self, cmd: str, stream_output: bool = False) -> int:
        self.connect()

        full_cmd = f"stdbuf -o0 bash -c {shlex.quote(cmd)}"
        stdin, stdout, stderr = self.ssh.exec_command(full_cmd, get_pty=True)

        if stream_output:
            for line in iter(stdout.readline, ""):
                print(line, end="")
            for line in iter(stderr.readline, ""):
                print(line, end="")
        else:
            output = stdout.read().decode("utf-8") + stderr.read().decode("utf-8")
            print(output)

        return stdout.channel.recv_exit_status()

    # prepare training data
    # rdf_pairs: list of tuples (mic_img_v_path, rdf_path) — local absolute paths
    # produces *_roi.v via thrass, then symlinks them into training staging dir.id
    def prepare_training_data(self, rdf_pairs, progress_callback=None):
        self.connect()

        original_dir = os.path.dirname(os.path.abspath(rdf_pairs[0][0]))
        retrain_dir = os.path.join(original_dir, "retrain")
        self._exec_local(f"mkdir -p {shlex.quote(retrain_dir)}")

        # symlink inputs
        for v, rdf in rdf_pairs:
            base = os.path.splitext(os.path.basename(v))[0]
            v_link = os.path.join(retrain_dir, f"{base}_img.v")
            rdf_link = os.path.join(retrain_dir, f"{base}_img.rdf")
            self._exec_local(f"ln -sf {shlex.quote(v)} {shlex.quote(v_link)}")
            self._exec_local(f"ln -sf {shlex.quote(rdf)} {shlex.quote(rdf_link)}")

        if progress_callback:
            progress_callback(0, 0, "symlinking", "rdf_to_masks")

        # produce *_roi.v via thrass
        self._exec_local(f"cd {shlex.quote(retrain_dir)} && bash {shlex.quote(self.preprocess_script)}")

        if progress_callback:
            progress_callback(0, 0, "thrass", "rdf_to_masks")

        # symlink results into training staging dir
        symlink_cmd = (
            f"cd {shlex.quote(self.training_staging_dir)} && "
            f"ln -sf {shlex.quote(retrain_dir)}/*_img.v {shlex.quote(retrain_dir)}/*_roi.v ."
        )
        self._exec_local(symlink_cmd)

        if progress_callback:
            progress_callback(0, 0, self.training_staging_dir, "linked_training")

        return {}

    # run createDataset.sh
    def create_dataset(self, dataset_id: int, description: str, progress_callback=None):
        ds = int(dataset_id)
        desc = shlex.quote(description)

        cmd = f"cd {shlex.quote(self.dataset_workdir)} && {shlex.quote(self.create_dataset_script)} {ds} {desc}"

        self._exec_local(cmd, stream_output=True)

        if progress_callback:
            progress_callback(0, 0, f"Dataset {ds}", "dataset_created")

    # nnUNet plan & preprocess
    def preprocess_dataset(self, dataset_id: int, progress_callback=None):
        ds = int(dataset_id)

        # nnUNet plan & preprocess
        pp_cmd = (
            f"{shlex.quote(self.conda_bin)} run -n {shlex.quote(self.nnunet_env)} --live-stream "
            f"nnUNetv2_plan_and_preprocess -d {ds} --verify_dataset_integrity -c {shlex.quote(self.config)} -np 8"
        )
        self._exec_remote(pp_cmd, stream_output=True)
        if progress_callback:
            progress_callback(0, 0, f"Dataset {ds}", "preprocessed")

    # create splits_final.json file and copy to preprocessed folder
    def create_custom_split(self, dataset_id: int, description: str, progress_callback=None):
        # paths
        images_dir = os.path.join(
            self.dataset_workdir, f"Dataset{dataset_id:03d}_spheroids_{description}", "imagesTr"
        )
        output_path = os.path.join(
            self.nnunet_preprocessed_dir, f"Dataset{dataset_id:03d}_spheroids_{description}", "splits_final.json"
        )

        # collect *_0000.v files
        files = [fn.replace("_0000.v", "") for fn in os.listdir(images_dir) if fn.endswith("_0000.v")]

        # unique spheroid IDs
        spheroid_ids = set(int(fn.split("_")[-1]) for fn in files)
        min_id, max_id = min(spheroid_ids), max(spheroid_ids)
        id_per_fold = int(max_id / 5)

        # split lists
        a, b, c, d, e = [], [], [], [], []
        for file in files:
            dataset_num = int(file.split('_')[-1])
            if min_id <= dataset_num <= id_per_fold:
                a.append(file)
            elif id_per_fold + 1 <= dataset_num <= id_per_fold * 2 + 1:
                b.append(file)
            elif id_per_fold * 2 + 2 <= dataset_num <= id_per_fold * 3 + 2:
                c.append(file)
            elif id_per_fold * 3 + 3 <= dataset_num <= id_per_fold * 4 + 3:
                d.append(file)
            elif id_per_fold * 4 + 4 <= dataset_num <= max_id:
                e.append(file)

        # create 5-fold splits
        splits = [
            {'train': a + b + c + d, 'val': e},
            {'train': a + b + c + e, 'val': d},
            {'train': a + b + d + e, 'val': c},
            {'train': a + c + d + e, 'val': b},
            {'train': b + c + d + e, 'val': a}
        ]

        # save splits_final.json
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(splits, f, indent=4)

        if progress_callback:
            progress_callback(0, 0, f"Dataset {dataset_id}", "split_done")

    # train folds (multi-GPU)
    def train_folds(
        self,
        dataset_id: int,
        folds: List[int],
        gpu_ids: List[int],
        progress_callback=None
    ):
        ds = int(dataset_id)
        if not folds:
            folds = [0, 1, 2, 3, 4]
        if not gpu_ids:
            gpu_ids = [0]

        def _train_one(gpu: int, fold: int):
            if progress_callback:
                progress_callback(fold, len(folds), f"fold={fold}, gpu={gpu}", "training_start")
            cmd = (
                f"CUDA_VISIBLE_DEVICES={gpu} "
                f"{shlex.quote(self.conda_bin)} run -n {shlex.quote(self.nnunet_env)} --live-stream "
                f"{shlex.quote(self.nnunet_train_bin)} -tr {shlex.quote(self.nnunet_trainer)} {ds} {shlex.quote(self.config)} {fold}"
            )
            self._exec_remote(cmd, stream_output=True)
            if progress_callback:
                progress_callback(fold + 1, len(folds), f"fold={fold}, gpu={gpu}", "training_finished")

        threads = []
        for i, f in enumerate(folds):
            gpu = gpu_ids[i % len(gpu_ids)]
            t = threading.Thread(target=_train_one, args=(gpu, f), daemon=True)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()

    def retrainCall(
        self,
        rdf_pairs: List[Tuple[str, str]],
        dataset_id: int,
        description: str,
        gpu_ids: Optional[List[int]] = None,
        folds: Optional[List[int]] = None,
        progress_callback=None
    ):
        try:
            # full pipeline
            self.prepare_training_data(rdf_pairs, progress_callback=progress_callback)
            self.create_dataset(dataset_id, description, progress_callback=progress_callback)
            #self.preprocess_dataset(dataset_id, progress_callback=progress_callback)
            #self.create_custom_split(dataset_id, description, progress_callback=progress_callback)
            """self.train_folds(
                dataset_id,
                folds or [0, 1, 2, 3, 4],
                gpu_ids or [0, 1, 2, 3],
                progress_callback=progress_callback
            )"""
        finally:
            self.disconnect()