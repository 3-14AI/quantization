import argparse
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from awq import AutoAWQForCausalLM

def load_and_format_calibration_data(parquet_path, tokenizer):
    print(f"Loading calibration data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    formatted_data = []
    for _, row in df.iterrows():
        messages = row.get("messages")
        format_output = row.get("format_output")

        if messages is None:
            continue

        # Handle numpy arrays to lists if needed
        if isinstance(messages, np.ndarray):
            messages = messages.tolist()

        if len(messages) == 0:
            continue

        # Apply chat template
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        # If there is an output, append it manually
        if pd.notna(format_output) and format_output:
             text += str(format_output)

        formatted_data.append(text)

    print(f"Loaded and formatted {len(formatted_data)} examples.")
    return formatted_data

def main():
    parser = argparse.ArgumentParser(description="Quantize T-lite-it-2.1 with AutoAWQ using custom parquet calibration data")
    parser.add_argument("--model_path", type=str, default="t-tech/T-lite-it-2.1", help="Path to the model or HF model ID")
    parser.add_argument("--calib_data", type=str, required=True, help="Path to the parquet calibration data file")
    parser.add_argument("--quant_path", type=str, default="T-lite-it-2.1-awq", help="Path to save the quantized model")
    parser.add_argument("--zero_point", type=bool, default=True, help="Use zero point for quantization")
    parser.add_argument("--q_group_size", type=int, default=128, help="Group size for quantization")
    parser.add_argument("--w_bit", type=int, default=4, help="Weight bit width")
    parser.add_argument("--version", type=str, default="GEMM", help="Quantization version")
    args = parser.parse_args()

    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    calib_texts = load_and_format_calibration_data(args.calib_data, tokenizer)

    quant_config = {
        "zero_point": args.zero_point,
        "q_group_size": args.q_group_size,
        "w_bit": args.w_bit,
        "version": args.version
    }

    print(f"Loading model from {args.model_path}...")
    model = AutoAWQForCausalLM.from_pretrained(args.model_path, trust_remote_code=True)

    print("Starting quantization...")
    # AutoAWQ's model.quantize() accepts `calib_data` which can be a list of strings
    model.quantize(tokenizer, quant_config=quant_config, calib_data=calib_texts)

    print(f"Saving quantized model to {args.quant_path}...")
    model.save_quantized(args.quant_path)
    tokenizer.save_pretrained(args.quant_path)
    print("Done!")

if __name__ == "__main__":
    main()
