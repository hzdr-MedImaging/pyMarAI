import sys
import argparse
from marai import MarAiLocal, MarAiRemote

# custom ArgParser for better GUI integration
class ArgParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write(f'ERROR: {message}\n\n')
        self.print_help()
        self.exit(1)

def main(args):
    parser = ArgParser(
        prog="marai-predict",
        description='Start prediction (inference) for microscopic images of spheroids',
        add_help=False
    )

    parser.add_argument("--input", action="append", required=True, help="Input files")
    parser.add_argument("--output", required=True, help="Directory to store output files")
    parser.add_argument("--microscope", required=True, help="Microscope ID used to obtain images")

    # mutually exclusive mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--local", action="store_true", help="Run prediction locally")
    mode_group.add_argument("--remote", action="store_true", help="Run prediction remotely (default)")

    parsed_args = parser.parse_args(args)

    run_analysis(
        input_files=parsed_args.input,
        microscope_number=parsed_args.microscope,
        output_dir=parsed_args.output,
        use_local=parsed_args.local
    )

def run_analysis(input_files, microscope_number, output_dir, use_local=False):

    predictor = MarAiLocal() if use_local else MarAiRemote()

    for input_file in input_files:

        try:
            print(f"[INFO] Processing {input_file} ...")
            predictor.predictSingleFile(
                input_file=input_file,
                microscope_number=microscope_number,
                output_dir=output_dir
            )
            print(f"[DONE] {input_file}")
        except Exception as e:
            print(f"[ERROR] Failed to process {input_file}: {e}")