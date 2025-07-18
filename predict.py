import sys
import argparse
import traceback
import logging
from marai import MarAiLocal, MarAiRemote
from multiprocessing import Event
from multiprocessing.connection import Connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# custom ArgParser for better GUI integration
class ArgParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write(f'ERROR: {message}\n\n')
        self.print_help()
        sys.exit(1)

def gui_entry_point(params: dict, username: str, password: str, hostname: str,
                    progress_pipe_connection: Connection, stdout_pipe_connection: Connection,
                    stop_event: Event = None):

    logger.info("predict.gui_entry_point received call from GUI process.")

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

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
                except Exception as e:
                    sys.stderr.write(f"Error sending log message to pipe: {e}\n")

    pipe_handler = PipeHandler(stdout_pipe_connection)
    root_logger.addHandler(pipe_handler)
    root_logger.setLevel(logging.INFO)

    def gui_progress_callback(current_count: int, total_count: int, filename: str):
        if progress_pipe_connection and not progress_pipe_connection.closed:
            try:
                progress_pipe_connection.send((current_count, total_count, filename))
            except Exception as e:
                logger.error(f"Error sending progress update through pipe: {e}")

    try:
        # Extract params for run_analysis
        input_files = params.get('input_files', [])
        output_dir = params.get('output_dir')
        microscope_number = params.get('microscope_number')

        if not input_files or not output_dir or microscope_number is None:
            raise ValueError("Missing required parameters for prediction.")

        use_local = (hostname is None or hostname.strip() == '')

        run_analysis(
            input_files=input_files,
            microscope_number=microscope_number,
            output_dir=output_dir,
            use_local=use_local,
            ssh_username=username,
            ssh_password=password,
            stop_event=stop_event,
            progress_callback=gui_progress_callback
        )
    except Exception as e:
        logger.error(f"[ERROR] An unhandled exception occurred in predict.gui_entry_point: {e}", exc_info=True)
        raise

    finally:
        root_logger.removeHandler(pipe_handler)

        if stdout_pipe_connection and not stdout_pipe_connection.closed:
            try:
                stdout_pipe_connection.close()
                logger.debug("Child stdout pipe connection closed in gui_entry_point.")
            except Exception as e:
                sys.stderr.write(f"Error closing stdout_pipe_connection in gui_entry_point: {e}\n")
        if progress_pipe_connection and not progress_pipe_connection.closed:
            try:
                progress_pipe_connection.close()
                logger.debug("Child progress pipe connection closed in gui_entry_point.")
            except Exception as e:
                sys.stderr.write(f"Error closing progress_pipe_connection in gui_entry_point: {e}\n")


def main(args=None):
    parser = ArgParser(
        prog="marai-predict",
        description='Start prediction (inference) for microscopic images of spheroids',
        add_help=True
    )

    parser.add_argument("--input", action="append", required=True, help="Input file(s) to process")
    parser.add_argument("--output", required=True, help="Directory to store output files")
    parser.add_argument("--microscope", type=int, required=True, help="Microscope ID used to obtain images")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--local", action="store_true", help="Run prediction locally")
    mode_group.add_argument("--remote", action="store_true", default=True,
                            help="Run prediction remotely (default)")

    parser.add_argument("--ssh-username", help="SSH username for remote connection")
    parser.add_argument("--ssh-password", help="SSH password for remote connection")

    parsed_args = parser.parse_args(args)

    cli_process_stop_event = Event()

    def cli_progress_callback(current_count: int, total_count: int, filename: str):
        logger.info(f"Progress: {current_count}/{total_count} files processed. Last: {filename}")

    try:
        run_analysis(
            input_files=parsed_args.input,
            microscope_number=parsed_args.microscope,
            output_dir=parsed_args.output,
            use_local=parsed_args.local,
            ssh_username=parsed_args.ssh_username,
            ssh_password=parsed_args.ssh_password,
            stop_event=cli_process_stop_event,
            progress_callback=cli_progress_callback
        )
    except Exception as e:
        logger.critical(f"CLI execution failed: {e}")
        sys.exit(1)
    sys.exit(0)

def run_analysis(input_files: list, microscope_number: int, output_dir: str,
                use_local: bool = False, ssh_username: str = None, ssh_password: str = None,
                stop_event: Event = None, progress_callback = None):
    logger.info(f"[INFO] run_analysis called with input_files={input_files}, microscope_number={microscope_number}, output_dir={output_dir}, use_local={use_local}")

    predictor = None
    try:
        if use_local:
            logger.info("[INFO] Using local predictor.")
            predictor = MarAiLocal(stop_event=stop_event)
        else:
            logger.info("[INFO] Using remote predictor.")
            predictor = MarAiRemote(username=ssh_username, password=ssh_password, stop_event=stop_event)

        logger.info(f"[INFO] Processing {len(input_files)} files...")
        predictor.predictCall(
            input_files=input_files,
            microscope_number=microscope_number,
            output_dir=output_dir,
            progress_callback=progress_callback
        )
        logger.info(f"[INFO] Processed {len(input_files)} files. DONE.")
    except Exception as e:
        logger.error(f"[ERROR] Failed to process files: {e}")
        logger.error(traceback.format_exc())
        raise
    finally:
        if isinstance(predictor, MarAiRemote):
            try:
                if hasattr(predictor, 'disconnect') and callable(predictor.disconnect):
                    predictor.disconnect()
            except Exception as e:
                logger.warning(f"Error during MarAiRemote disconnect: {e}")


if __name__ == "__main__":
    main()