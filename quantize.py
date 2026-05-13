import json
import click
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union, Optional
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from awq import AutoAWQForCausalLM
from huggingface_hub import HfApi

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
@click.option("--w_bit", type=int, default=4, help="Weight bit width")
@click.option("--version", type=str, default="GEMM", help="Quantization version")
@click.option("--push_to_hub", is_flag=True, help="Whether to push the quantized model to the Hugging Face Hub")
@click.option("--hub_repo_id", type=str, default=None, help="The Hugging Face Hub repository ID to push to (e.g., 'username/model-name')")
def main(
    model_path: str,
    calib_data: str,
    quant_path: str,
    zero_point: bool,
    q_group_size: int,
    w_bit: int,
    version: str,
    push_to_hub: bool,
    hub_repo_id: Optional[str]
) -> None:
    """Quantize T-lite-it-2.1 with AutoAWQ using custom parquet calibration data."""

    print(f"Loading tokenizer from {model_path}...")
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    calib_texts: List[str] = load_and_format_calibration_data(calib_data, tokenizer)

    quant_config: Dict[str, Any] = {
        "zero_point": zero_point,
        "q_group_size": q_group_size,
        "w_bit": w_bit,
        "version": version
    }

    print(f"Loading model from {model_path}...")
    model = AutoAWQForCausalLM.from_pretrained(model_path, trust_remote_code=True)

    print("Starting quantization...")
    # AutoAWQ's model.quantize() accepts `calib_data` which can be a list of strings
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_texts)

    print(f"Saving quantized model to {quant_path}...")
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)
    print("Done!")

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
