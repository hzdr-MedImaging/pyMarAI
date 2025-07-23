import sys
import os
import argparse
import traceback
import logging
from marai import MarAiLocal, MarAiRemote
from multiprocessing import Event
from multiprocessing.connection import Connection

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def make_progress_callback(log_fn):
    def callback(current_count: int, total_count: int, filename: str, stage_indicator: str):
        base_name = os.path.basename(filename)
        if stage_indicator == "nnunet_predicting":
            log_fn(f"NNUNet Progress: {current_count}/{total_count} files. Currently processing: {base_name}")
        elif stage_indicator == "rdf_finished":
            log_fn(f"RDF Conversion Progress: {current_count}/{total_count} files. Finished: {base_name}")
        else:
            log_fn(f"Progress: {current_count}/{total_count} files processed. Last: {base_name}")
    return callback

def gui_entry_point(params: dict, username: str, password: str,
                    progress_pipe_connection: Connection, stdout_pipe_connection: Connection,
                    stop_event: Event = None) -> None:

    logger.info("predict.gui_entry_point received call from GUI process.")

    root_logger_child_process = logging.getLogger()
    for handler in root_logger_child_process.handlers[:]:
        root_logger_child_process.removeHandler(handler)

    class PipeHandler(logging.Handler):
        def __init__(self, pipe_conn: Connection):
            super().__init__()
            self.pipe = pipe_conn
            self.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.setLevel(logging.DEBUG)

        def emit(self, record):
            if self.pipe and not self.pipe.closed:
                try:
                    msg = self.format(record)
                    self.pipe.send(msg + "\n")
                except Exception as e:
                    sys.stderr.write(f"Error sending log message to pipe: {e}\n")

    pipe_handler = PipeHandler(stdout_pipe_connection)
    root_logger_child_process.addHandler(pipe_handler)
    root_logger_child_process.setLevel(logging.DEBUG)

    progress_callback = make_progress_callback(logger.info)

    try:
        input_files = params.get('input_files', [])
        output_dir = params.get('output_dir')
        microscope_number = params.get('microscope_number')

        if not input_files or not output_dir or microscope_number is None:
            raise ValueError("Missing required parameters for prediction.")

        use_local = (not username or not password)

        run_analysis(
            input_files=input_files,
            microscope_number=microscope_number,
            output_dir=output_dir,
            use_local=use_local,
            ssh_username=username,
            ssh_password=password,
            stop_event=stop_event,
            progress_callback=progress_callback
        )
    except Exception as e:
        logger.error(f"An unhandled exception occurred in predict.gui_entry_point: {e}", exc_info=True)
        raise
    finally:
        root_logger_child_process.removeHandler(pipe_handler)
        logger.debug("PipeHandler removed from root logger in gui_entry_point.")

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


def main(args=None) -> None:
    class ArgParser(argparse.ArgumentParser):
        def error(self, message):
            sys.stderr.write(f'ERROR: {message}\n\n')
            self.print_help()
            sys.exit(1)

    parser = ArgParser(
        prog="marai-predict",
        description='Start prediction (inference) for microscopic images of spheroids',
        add_help=True
    )

    parser.add_argument("--input", action="append", required=True, help="Input file(s) to process")
    parser.add_argument("--output", required=True, help="Directory to store output files")
    parser.add_argument("--microscope", type=int, required=True, help="Microscope ID used to obtain images")

    parser.add_argument("--local", action="store_true", help="Run prediction locally")
    parser.add_argument("--ssh-username", help="SSH username for remote connection")
    parser.add_argument("--ssh-password", help="SSH password for remote connection")

    parsed_args = parser.parse_args(args)

    progress_callback = make_progress_callback(logger.info)
    cli_process_stop_event = Event()

    try:
        run_analysis(
            input_files=parsed_args.input,
            microscope_number=parsed_args.microscope,
            output_dir=parsed_args.output,
            use_local=parsed_args.local,
            ssh_username=parsed_args.ssh_username,
            ssh_password=parsed_args.ssh_password,
            stop_event=cli_process_stop_event,
            progress_callback=progress_callback
        )
    except Exception as e:
        logger.critical(f"CLI execution failed: {e}", exc_info=True)
        sys.exit(1)
    sys.exit(0)


def run_analysis(input_files: list, microscope_number: int, output_dir: str,
                 use_local: bool = False, ssh_username: str = None, ssh_password: str = None,
                 stop_event: Event = None, progress_callback=None) -> None:
    logger.info(f"run_analysis called with input_files={input_files}, microscope_number={microscope_number}, output_dir={output_dir}, use_local={use_local}")

    predictor = None
    try:
        if use_local:
            logger.info("Using local predictor.")
            predictor = MarAiLocal(stop_event=stop_event)
        else:
            logger.info("Using remote predictor.")
            predictor = MarAiRemote(username=ssh_username, password=ssh_password, stop_event=stop_event)

        logger.info(f"Processing {len(input_files)} files...")
        predictor.predictCall(
            input_files=input_files,
            microscope_number=microscope_number,
            output_dir=output_dir,
            progress_callback=progress_callback
        )
        logger.info(f"Processed {len(input_files)} files. DONE.")
    except Exception as e:
        logger.error(f"Failed to process files: {e}")
        logger.error(traceback.format_exc())
        raise
    finally:
        if isinstance(predictor, MarAiRemote):
            try:
                if hasattr(predictor, 'disconnect') and callable(predictor.disconnect) and predictor.connected:
                    predictor.disconnect()
            except Exception as e:
                logger.warning(f"Error during MarAiRemote disconnect: {e}")

if __name__ == "__main__":
    main()