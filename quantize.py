import json
import click
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union, Optional
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from awq import AutoAWQForCausalLM
from huggingface_hub import HfApi

import os
import subprocess
import shutil
import tempfile
from huggingface_hub import snapshot_download


def load_and_format_calibration_data(parquet_path: str, tokenizer: PreTrainedTokenizerBase) -> List[str]:
    print(f"Loading calibration data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    formatted_data: List[str] = []
    for _, row in df.iterrows():
        request_str: str = row.get("request")
        response_str: str = row.get("response")

        if pd.isna(request_str) or not request_str:
            continue

        try:
            request = json.loads(request_str)
        except json.JSONDecodeError:
            continue

        messages = request.get("messages", [])
        if not messages:
            continue

        if pd.notna(response_str) and response_str:
            try:
                response = json.loads(response_str)
                if response and "choices" in response and len(response["choices"]) > 0:
                    messages.append(response["choices"][0]["message"])
            except json.JSONDecodeError:
                pass

        kwargs = {}
        if "tools" in request:
            kwargs["tools"] = request["tools"]
        if "response_format" in request:
            kwargs["response_format"] = request["response_format"]

        try:
            text: str = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                **kwargs
            )
            formatted_data.append(text)
        except Exception as e:
            print(f"Warning: Failed to format conversation: {e}")
            continue

    print(f"Loaded and formatted {len(formatted_data)} examples.")
    return formatted_data


@click.command()
@click.option("--model_path", type=str, default="t-tech/T-lite-it-2.1", help="Path to the model or HF model ID")
@click.option("--calib_data", type=str, required=True, help="Path to the parquet calibration data file")
@click.option("--quant_path", type=str, default="T-lite-it-2.1-awq", help="Path to save the quantized model")
@click.option("--zero_point", type=bool, default=True, help="Use zero point for quantization")
@click.option("--q_group_size", type=int, default=128, help="Group size for quantization")
@click.option("--method", type=click.Choice(['awq', 'exl2']), default="awq", help="Quantization method to use")
@click.option("--w_bit", type=int, default=4, help="Weight bit width (AWQ)")
@click.option("--version", type=str, default="GEMM", help="Quantization version (AWQ)")
@click.option("--exl2_bits", type=float, default=4.0, help="Target bits per weight (EXL2)")
@click.option("--exl2_head_bits", type=int, default=6, help="Target bits per weight for head layer (EXL2)")
@click.option("--push_to_hub", is_flag=True, help="Whether to push the quantized model to the Hugging Face Hub")
@click.option("--hub_repo_id", type=str, default=None, help="The Hugging Face Hub repository ID to push to (e.g., 'username/model-name')")
def main(
    model_path: str,
    calib_data: str,
    quant_path: str,
    method: str,
    zero_point: bool,
    q_group_size: int,
    w_bit: int,
    version: str,
    exl2_bits: float,
    exl2_head_bits: int,
    push_to_hub: bool,
    hub_repo_id: Optional[str]
) -> None:
    """Quantize a model using AutoAWQ or EXL2 with custom parquet calibration data."""

    print(f"Loading tokenizer from {model_path}...")
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    calib_texts: List[str] = load_and_format_calibration_data(calib_data, tokenizer)

    if method == "awq":
        quant_config: Dict[str, Any] = {
            "zero_point": zero_point,
            "q_group_size": q_group_size,
            "w_bit": w_bit,
            "version": version
        }

        print(f"Loading model from {model_path}...")
        model = AutoAWQForCausalLM.from_pretrained(model_path, trust_remote_code=True)

        print("Starting AWQ quantization...")
        model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_texts)

        print(f"Saving quantized model to {quant_path}...")
        model.save_quantized(quant_path)
        tokenizer.save_pretrained(quant_path)
        print("AWQ quantization done!")
    elif method == "exl2":
        print("Starting EXL2 quantization...")

        # Download model if it's a hub path
        local_model_path = model_path
        if not os.path.isdir(model_path):
            print(f"Downloading model {model_path} from Hugging Face Hub...")
            local_model_path = snapshot_download(repo_id=model_path)

        # EXL2 expects a parquet file. We'll save the formatted text as a single column.
        # convert_exl2.py reads 'concatenated' or just encodes the rows.
        # We will write out the formatted data to a temporary parquet file.
        temp_dir = tempfile.mkdtemp()
        exl2_calib_parquet = os.path.join(temp_dir, "exl2_calib.parquet")

        df_calib = pd.DataFrame({"text": calib_texts})
        df_calib.to_parquet(exl2_calib_parquet)
        print(f"Saved EXL2 formatted calibration data to {exl2_calib_parquet}")

        # Make a working directory for EXL2 measurement and intermediate files
        exl2_work_dir = os.path.join(temp_dir, "exl2_work")
        os.makedirs(exl2_work_dir, exist_ok=True)
        os.makedirs(quant_path, exist_ok=True)

        # Construct EXL2 command
        # Note: EXL2 uses an environment variable EXLLAMA_NO_COMPILE=1 if compiling fails
        env = os.environ.copy()
        env["EXLLAMA_NO_COMPILE"] = "1"

        # EXL2 conversion script is usually downloaded from the exllamav2 repo.
        # But convert_exl2 actually exists inside exllamav2 package. Let's make sure we find its exact path
        # Import exllamav2 might fail if CUDA_HOME is not set, so we can construct path dynamically
        # or use EXLLAMA_NO_COMPILE=1 for importing
        os.environ["EXLLAMA_NO_COMPILE"] = "1"
        try:
            import exllamav2
            exl2_convert_script = os.path.join(os.path.dirname(exllamav2.__file__), "conversion", "convert_exl2.py")
        except ImportError as e:
            print(f"Warning: Could not import exllamav2. Finding path manually. Error: {e}")
            import site
            site_packages = site.getsitepackages()
            for sp in site_packages:
                p = os.path.join(sp, "exllamav2", "conversion", "convert_exl2.py")
                if os.path.exists(p):
                    exl2_convert_script = p
                    break
            else:
                raise RuntimeError("Could not find exllamav2 conversion script convert_exl2.py")

        cmd = [
            "python", exl2_convert_script,
            "-i", local_model_path,
            "-o", exl2_work_dir,
            "-cf", quant_path,
            "-c", exl2_calib_parquet,
            "-b", str(exl2_bits),
            "-hb", str(exl2_head_bits)
        ]

        print(f"Running EXL2 conversion: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True, env=env)
            print("EXL2 quantization done!")
        except subprocess.CalledProcessError as e:
            print(f"Error during EXL2 quantization: {e}")
            raise
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir, ignore_errors=True)

    if push_to_hub:
        if not hub_repo_id:
            print("Error: --hub_repo_id must be provided if --push_to_hub is used.")
            return
        print(f"Pushing to Hugging Face Hub: {hub_repo_id}...")
        api = HfApi()
        api.upload_folder(
            folder_path=quant_path,
            repo_id=hub_repo_id,
            repo_type="model",
        )
        print("Successfully pushed to Hugging Face Hub!")

if __name__ == "__main__":
    main()
