import click
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from awq import AutoAWQForCausalLM

def load_and_format_calibration_data(parquet_path: str, tokenizer: PreTrainedTokenizerBase) -> List[str]:
    print(f"Loading calibration data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    formatted_data: List[str] = []
    for _, row in df.iterrows():
        messages: Union[List[Dict[str, Any]], np.ndarray, None] = row.get("messages")
        format_output: Any = row.get("format_output")

        if messages is None:
            continue

        # Handle numpy arrays to lists if needed
        if isinstance(messages, np.ndarray):
            messages = messages.tolist()

        if len(messages) == 0:
            continue

        # Apply chat template
        text: str = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        # If there is an output, append it manually
        if pd.notna(format_output) and format_output:
             text += str(format_output)

        formatted_data.append(text)

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
def main(
    model_path: str,
    calib_data: str,
    quant_path: str,
    zero_point: bool,
    q_group_size: int,
    w_bit: int,
    version: str
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

if __name__ == "__main__":
    main()
