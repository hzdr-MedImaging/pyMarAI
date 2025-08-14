import sys
import os
import argparse
import traceback
import logging
import platform
from pymarai.marai import MarAiLocal, MarAiRemote
from pymarai.config import AppConfig
from multiprocessing import Event
from multiprocessing.connection import Connection

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- core task ---
class PredictionTask:
    def __init__(self, input_files: list, output_dir: str, microscope_number: int, use_local: bool = True,
                 ssh_username: str = None, ssh_password: str = None, ssh_keys: list = None, hostname: str = None,
                 gpu_id: str = None, stop_event: Event = None, progress_callback=None, log_fn=None):
        self.input_files = input_files
        self.output_dir = output_dir
        self.microscope_number = microscope_number
        self.use_local = use_local
        self.ssh_username = ssh_username
        self.ssh_password = ssh_password
        self.ssh_keys = ssh_keys
        self.hostname = hostname
        self.gpu_id = gpu_id
        self.stop_event = stop_event
        self.progress_callback = progress_callback
        self.log_fn = log_fn or logger.info
        self.predictor = None

    def run(self):
        self.log_fn(f"Starting prediction for {len(self.input_files)} file(s)...")
        try:
            self.predictor = self._get_predictor()
            self.predictor.predictCall(
                input_files=self.input_files,
                microscope_number=self.microscope_number,
                output_dir=self.output_dir,
                progress_callback=self.progress_callback
            )
            self.log_fn(f"Prediction complete for {len(self.input_files)} file(s).")
        except Exception as e:
            self.log_fn(f"[ERROR] Prediction failed: {e}")
            self.log_fn(traceback.format_exc())
            raise
        finally:
            self._disconnect_if_needed()

    def _get_predictor(self):
        if self.use_local:
            self.log_fn("Using local predictor.")
            return MarAiLocal(stop_event=self.stop_event)
        else:
            self.log_fn(f"Using remote predictor on host: {self.hostname}.")
            return MarAiRemote(
                hostname=self.hostname,
                username=self.ssh_username,
                password=self.ssh_password,
                ssh_keys=self.ssh_keys,
                stop_event=self.stop_event,
                gpu_id=self.gpu_id
            )

    def _disconnect_if_needed(self):
        if isinstance(self.predictor, MarAiRemote):
            try:
                if getattr(self.predictor, 'connected', False):
                    self.predictor.disconnect()
                    self.log_fn("Remote predictor disconnected.")
            except Exception as e:
                self.log_fn(f"Warning: error during disconnect: {e}")


# --- progress callback factory ---
def make_progress_callback(progress_conn: Connection = None, log_fn=print):
    def callback(current_count: int, total_count: int, filename: str, stage_indicator: str):
        base_name = os.path.basename(filename)
        if stage_indicator == "nnunet_predicting":
            msg = f"NNUNet Progress: {current_count}/{total_count} files. Currently processing: {base_name}"
        elif stage_indicator == "rdf_finished":
            msg = f"RDF Conversion Progress: {current_count}/{total_count} files. Finished: {base_name}"
        else:
            msg = f"Progress: {current_count}/{total_count} files processed. Last: {base_name}"

        if log_fn:
            log_fn(msg)

        if progress_conn:
            try:
                progress_conn.send((current_count, total_count, filename, stage_indicator))
            except Exception as e:
                if log_fn:
                    log_fn(f"[ERROR] Could not send progress to pipe: {e}")

    return callback


# --- GUI entry point ---
def gui_entry_point(params: dict, username: str, password: str, ssh_keys: list,
                    progress_pipe_connection: Connection,
                    stdout_pipe_connection: Connection,
                    stop_event: Event = None):

    # Setup logging to send to stdout_pipe
    class PipeHandler(logging.Handler):
        def __init__(self, pipe_conn: Connection):
            super().__init__()
            self.pipe = pipe_conn
            self.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        def emit(self, record):
            if self.pipe and not self.pipe.closed:
                try:
                    msg = self.format(record)
                    self.pipe.send(msg + "\n")
                except Exception:
                    pass

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    pipe_handler = PipeHandler(stdout_pipe_connection)
    root_logger.addHandler(pipe_handler)
    root_logger.setLevel(logging.DEBUG)

    def gui_log_fn(msg):
        root_logger.debug(msg)

    progress_callback = make_progress_callback(progress_pipe_connection, gui_log_fn)

    try:
        hostname, gpu_id = AppConfig().get_best_available_host()
        if not hostname:
            raise RuntimeError("No available host found.")

        current_hostname = platform.node()
        use_local = (current_hostname == hostname)

        if use_local:
            gui_log_fn(f"Selected local host {hostname}. Running locally.")
        elif username and not password and ssh_keys:
            gui_log_fn(f"Running remotely on {hostname} using SSH keys (no password).")
        elif username and password and not ssh_keys:
            gui_log_fn(f"Running remotely on {hostname} using password authentication.")

        if gpu_id:
            gui_log_fn(f"Performing GPU processing using GPU {gpu_id} on {hostname}")
        else:
            gui_log_fn(f"Performing CPU processing on {hostname}")

        task = PredictionTask(
            input_files=params.get('input_files', []),
            output_dir=params.get('output_dir'),
            microscope_number=params.get('microscope_number'),
            use_local=use_local,
            ssh_username=username,
            ssh_password=password,
            ssh_keys=ssh_keys,
            hostname=hostname,
            gpu_id=gpu_id,
            stop_event=stop_event,
            progress_callback=progress_callback,
            log_fn=gui_log_fn
        )
        task.run()

    except Exception as e:
        root_logger.error(f"Unhandled exception in gui_entry_point: {e}", exc_info=True)
        raise

    finally:
        # Clean up pipes
        if stdout_pipe_connection and not stdout_pipe_connection.closed:
            stdout_pipe_connection.close()
        if progress_pipe_connection and not progress_pipe_connection.closed:
            progress_pipe_connection.close()


# --- CLI entry point ---
def main(args=None):
    class ArgParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write(f"ERROR: {message}\n\n")
            self.print_help()
            sys.exit(1)

    parser = ArgParser(
        prog="marai-predict",
        description='Start prediction (inference) for microscopic images of spheroids',
        add_help=True
    )

    parser.add_argument("--input", action="append", required=True, help="Input file(s) to process")
    parser.add_argument("--output", required=True, help="Directory to store output files")
    parser.add_argument("--microscope", type=int, required=True, help="Microscope ID used")
    parser.add_argument("--local", action="store_true", help="Run prediction locally")
    parser.add_argument("--ssh-username", help="SSH username for remote connection")
    parser.add_argument("--ssh-password", help="SSH password for remote connection")

    parsed_args = parser.parse_args(args)

    progress_callback = make_progress_callback(log_fn=logger.info)
    cli_stop_event = Event()

    try:
        task = PredictionTask(
            input_files=parsed_args.input,
            output_dir=parsed_args.output,
            microscope_number=parsed_args.microscope,
            use_local=parsed_args.local,
            ssh_username=parsed_args.ssh_username,
            ssh_password=parsed_args.ssh_password,
            stop_event=cli_stop_event,
            progress_callback=progress_callback,
            log_fn=logger.info
        )
        task.run()
    except Exception as e:
        logger.critical(f"CLI execution failed: {e}", exc_info=True)
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()

