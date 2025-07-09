import os
import sys
import shutil
import subprocess
import tempfile
import argparse

def run_analysis(input_files, microscope_number, output_dir):
    mic2ecat_path = "/usr/local/petlib/bin/mic2ecat"
    roi2rdf_path = "/usr/local/petlib/bin/roi2Rdf"
    conda_env = "nnunet-spheroids"

    nnunet_cmd_template = (
        "conda run -n {env} nnUNetv2_predict "
        "-d Dataset035_spheroids_clean "
        "-i {input_dir} "
        "-o {output_dir} "
        "-f 0 1 2 3 4 "
        "-tr nnUNetTrainer_noSmooth "
        "-c 2d "
        "-p nnUNetPlans "
        "-device cpu"
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        nnunet_input_dir = os.path.join(temp_dir, "nnunet_input")
        nnunet_output_dir = os.path.join(temp_dir, "nnunet_output")
        os.makedirs(nnunet_input_dir, exist_ok=True)
        os.makedirs(nnunet_output_dir, exist_ok=True)

        for input_file in input_files:
            try:
                # get base name
                base_name = os.path.basename(input_file)
                file_prefix = os.path.splitext(base_name)[0]

                # copy .tif to temp dir
                temp_input_path = shutil.copy(input_file, temp_dir)

                # run mic2ecat
                subprocess.run(
                    [mic2ecat_path, "-j", str(microscope_number), temp_input_path],
                    check=True
                )

                # mic2ecat output: *.v
                mic2ecat_v_path = os.path.join(temp_dir, f"{file_prefix}.v")
                if not os.path.exists(mic2ecat_v_path):
                    print(f"[ERROR] mic2ecat output not found: {mic2ecat_v_path}")
                    continue

                # rename it to *_0000.v for nnUNet
                nnunet_input_v = os.path.join(nnunet_input_dir, f"{file_prefix}_0000.v")
                shutil.copy(mic2ecat_v_path, nnunet_input_v)

                # run nnUNetv2
                nnunet_command = nnunet_cmd_template.format(
                    env=conda_env,
                    input_dir=nnunet_input_dir,
                    output_dir=nnunet_output_dir
                )
                subprocess.run(
                    nnunet_command,
                    check=True,
                    shell=True,
                    executable='/bin/bash',
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )

                # nnUNet output: *.v (without _0000)
                nnunet_output_v = os.path.join(nnunet_output_dir, f"{file_prefix}.v")
                if not os.path.exists(nnunet_output_v):
                    print(f"[ERROR] nnUNet did not produce expected .v file: {nnunet_output_v}")
                    continue

                # run roi2Rdf on nnUNet output .v (no output path)
                subprocess.run(
                    [roi2rdf_path, nnunet_output_v],
                    check=True
                )

                # roi2Rdf creates .rdf in same dir as nnunet_output_v
                rdf_output_path = os.path.splitext(nnunet_output_v)[0] + ".rdf"

                if not os.path.exists(rdf_output_path):
                    print(f"[ERROR] Expected .rdf file not created: {rdf_output_path}")
                    continue

                # save mic2ecat result (original .v) to output dir
                final_v_path = os.path.join(output_dir, f"{file_prefix}.v")
                shutil.copy(mic2ecat_v_path, final_v_path)

                # copy .rdf to output dir
                final_rdf_path = os.path.join(output_dir, f"{file_prefix}.rdf")
                shutil.copy(rdf_output_path, final_rdf_path)

                print(f"Done: {file_prefix}")

            except subprocess.CalledProcessError as e:
                print(f"[ERROR] Subprocess failed for {input_file}: {e}")
            except Exception as e:
                print(f"[ERROR] Unexpected error for {input_file}: {e}")

def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--microscope", required=True)

    parsed_args = parser.parse_args(args)

    run_analysis(parsed_args.input, parsed_args.microscope, parsed_args.output)