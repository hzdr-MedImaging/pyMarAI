import sys
import argparse
import traceback
from marai import MarAiLocal, MarAiRemote

# custom ArgParser for better GUI integration
class ArgParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write(f'ERROR: {message}\n\n')
        self.print_help()
        self.exit(1)

def main(args=None):

    parser = ArgParser(
        prog="marai-predict",
        description='Start prediction (inference) for microscopic images of spheroids',
        add_help=True
    )

    parser.add_argument("--input", action="append", required=True, help="Input file(s) to process")
    parser.add_argument("--output", required=True, help="Directory to store output files")
    parser.add_argument("--microscope", required=True, help="Microscope ID used to obtain images")

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--local", action="store_true", help="Run prediction locally")
    mode_group.add_argument("--remote", action="store_true", help="Run prediction remotely (default)")

    # optional SSH credentials for remote (not passed via command line usually, but placeholder)
    parser.add_argument("--ssh-username", help="SSH username for remote connection")
    parser.add_argument("--ssh-password", help="SSH password for remote connection")

    parsed_args = parser.parse_args(args)

    run_analysis(
        input_files=parsed_args.input,
        microscope_number=parsed_args.microscope,
        output_dir=parsed_args.output,
        use_local=parsed_args.local,
        ssh_username=parsed_args.ssh_username,
        ssh_password=parsed_args.ssh_password
    )

def run_analysis(input_files, microscope_number, output_dir,
                use_local=False, ssh_username=None, ssh_password=None,
                log_fn=print):

    log_fn(f"[INFO] run_analysis called with input_files={input_files}, microscope_number={microscope_number}, output_dir={output_dir}, use_local={use_local}\n")

    if use_local:
        log_fn("[INFO] Using local predictor\n")
        predictor = MarAiLocal()
    else:
        log_fn("[INFO] Using remote predictor\n")
        predictor = MarAiRemote(username=ssh_username, password=ssh_password)

    try:
        log_fn(f"[INFO] Processing {len(input_files)} files ...\n")
        predictor.predictCall(
            input_files=input_files,
            microscope_number=microscope_number,
            output_dir=output_dir,
        )
        log_fn(f"[DONE] Processed {len(input_files)} files.\n")
    except Exception as e:
        log_fn(f"[ERROR] Failed to process files: {e}\n")
        log_fn(traceback.format_exc())

if __name__ == "__main__":
    main()