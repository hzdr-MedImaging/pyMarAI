import os
import argparse
import traceback
import logging
from typing import List, Tuple, Optional, Dict

from remarai import MarAiRemoteRetrain

from multiprocessing import Event
from multiprocessing.connection import Connection

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# scan a directory for *.v files and match them with *.rdf using the same basename
def collect_pairs_from_dir(data_dir: str) -> List[Tuple[str, str]]:

    if not os.path.isdir(data_dir):
        raise ValueError(f"--data-dir {data_dir} is not a directory.")

    # index all .rdf files by basename for fast lookup
    rdf_index: Dict[str, str] = {}
    for name in os.listdir(data_dir):
        if name.lower().endswith(".rdf"):
            base = os.path.splitext(name)[0]
            rdf_index[base] = os.path.abspath(os.path.join(data_dir, name))

    pairs: List[Tuple[str, str]] = []
    missing = 0
    for name in os.listdir(data_dir):
        if name.lower().endswith(".v"):
            base = os.path.splitext(name)[0]
            v_path = os.path.abspath(os.path.join(data_dir, name))
            rdf_path = rdf_index.get(base)
            if rdf_path and os.path.isfile(v_path):
                pairs.append((v_path, rdf_path))
            else:
                missing += 1
                logger.warning(f"Missing matching .rdf for {v_path}")

    # sort for deterministic ordering
    pairs.sort(key=lambda t: t[0])
    logger.info(f"Found {len(pairs)} matching .v/.rdf pairs in {data_dir}"
                + (f" ({missing} without .rdf)" if missing else ""))

    return pairs


# --- core task ---
class TrainingTask:
    def __init__(
        self,
        rdf_pairs: List[Tuple[str, str]],
        dataset_id: int,
        description: str,
        hostname: str,
        ssh_username: Optional[str] = None,
        ssh_password: Optional[str] = None,
        ssh_keys: Optional[List[str]] = None,
        gpu_ids: Optional[List[int]] = None,
        folds: Optional[List[int]] = None,
        stop_event: Optional[Event] = None,
        progress_callback=None,
        log_fn=None
    ):
        self.rdf_pairs = rdf_pairs
        self.dataset_id = dataset_id
        self.description = description
        self.hostname = hostname
        self.ssh_username = ssh_username
        self.ssh_password = ssh_password
        self.ssh_keys = ssh_keys or []
        self.gpu_ids = gpu_ids or [0, 1, 2, 3]
        self.folds = folds or [0, 1, 2, 3, 4]
        self.stop_event = stop_event
        self.progress_callback = progress_callback
        self.log_fn = log_fn or logger.info
        self.worker: Optional[MarAiRemoteRetrain] = None

    def run(self):
        self.log_fn(f"Starting (re-)training for dataset {self.dataset_id} "
                    f"with {len(self.rdf_pairs)} annotated images...")
        try:
            self.worker = MarAiRemoteRetrain(
                hostname=self.hostname,
                username=self.ssh_username,
                password=self.ssh_password,
                ssh_keys=self.ssh_keys,
                stop_event=self.stop_event
            )
            self.worker.retrainCall(
                rdf_pairs=self.rdf_pairs,
                dataset_id=self.dataset_id,
                description=self.description,
                gpu_ids=self.gpu_ids,
                folds=self.folds,
                progress_callback=self.progress_callback
            )
            self.log_fn("Training complete.")
        except Exception as e:
            self.log_fn(f"[ERROR] Training failed: {e}")
            self.log_fn(traceback.format_exc())
            raise


# --- progress callback ---
def make_progress_callback(progress_conn: Connection = None, log_fn=print):
    def callback(current: int, total: int, label: str, stage: str):
        stage_map = {
            # sync with MarAiRemoteRetrain
            "symlinking": "Symlinking inputs",
            "thrass": "RDF→Mask conversion",
            "linked_training": "Staged training data",
            "dataset_created": "Dataset created",
            "preprocessed": "Planned & preprocessed",
            "split_done": "Custom split ready",
            "training_start": "Training fold started",
            "training_finished": "Training fold finished",
        }
        msg = f"{stage_map.get(stage, stage)}: {label} ({current}/{total})"
        if log_fn:
            log_fn(msg)
        if progress_conn:
            try:
                progress_conn.send((current, total, label, stage))
            except (BrokenPipeError, EOFError, OSError):
                if log_fn:
                    log_fn("Progress connection closed; cannot send update.")
    return callback


# --- CLI entry point ---
def main(args=None):
    parser = argparse.ArgumentParser(
        prog="marai-retrain",
        description="Run end-to-end (re-)training on a remote GPU node."
    )
    parser.add_argument("--host", required=True, help="Remote hostname")
    parser.add_argument("--dataset", type=int, required=True, help="Dataset ID")
    parser.add_argument("--desc", required=True, help="Dataset description")
    parser.add_argument("--data-dir", required=True,
                        help="Directory containing .v and .rdf files (matched by basename)")
    parser.add_argument("--gpu", action="append", type=int,
                        help="GPU id(s) to use. Default: 0,1,2,3")
    parser.add_argument("--fold", action="append", type=int,
                        help="Fold(s) to train. Default: 0,1,2,3,4")
    parser.add_argument("--ssh-username")
    parser.add_argument("--ssh-password")
    parser.add_argument("--ssh-key", action="append",
                        help="Path to an SSH private key") # optional SSH keys
    args = parser.parse_args(args)

    data_dir = os.path.abspath(args.data_dir)
    rdf_pairs = collect_pairs_from_dir(data_dir)
    if not rdf_pairs:
        parser.error(f"No matching .v/.rdf pairs found in {data_dir}.")

    gpu_ids = args.gpu if args.gpu else [0, 1, 2, 3]
    folds = args.fold if args.fold else [0, 1, 2, 3, 4]

    cb = make_progress_callback(log_fn=logger.info)
    task = TrainingTask(
        rdf_pairs=rdf_pairs,
        dataset_id=args.dataset,
        description=args.desc,
        hostname=args.host,
        ssh_username=args.ssh_username,
        ssh_password=args.ssh_password,
        ssh_keys=args.ssh_key or [],
        gpu_ids=gpu_ids,
        folds=folds,
        progress_callback=cb,
        log_fn=logger.info
    )
    task.run()


if __name__ == "__main__":
    main()

