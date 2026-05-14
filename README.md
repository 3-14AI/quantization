# quantization

This project provides a script `quantize.py` to quantize Hugging Face language models using AutoAWQ or EXL2 with a custom Parquet dataset for calibration.

## Requirements
Use `uv` to manage dependencies.

## Usage

You can run the script via `click` CLI:

```bash
uv run python quantize.py --calib_data path/to/calib.parquet [OPTIONS]
```

### Required Parameters

* `--calib_data`: (String) Path to the parquet calibration data file containing requests and responses to format.

### Optional General Parameters

* `--model_path`: (String) Path to the local model directory or Hugging Face model ID. Default: `t-tech/T-lite-it-2.1`.
* `--quant_path`: (String) Output directory where the quantized model will be saved. Default: `T-lite-it-2.1-awq`.
* `--method`: (Choice: `awq` or `exl2`) Specifies the quantization backend to use. Default: `awq`.
* `--push_to_hub`: (Flag) Set this flag to push the quantized model to the Hugging Face Hub after quantization.
* `--hub_repo_id`: (String) The Hugging Face Hub repository ID to push to (e.g., `'username/model-name'`). Required if `--push_to_hub` is used.

### AWQ-Specific Parameters (used when `--method=awq`)

* `--w_bit`: (Integer) Target weight bit width (e.g., 4 or 8). Default: `4`.
* `--q_group_size`: (Integer) Group size for quantization. Default: `128`.
* `--zero_point`: (Boolean) Whether to use zero-point quantization. Default: `True`.
* `--version`: (String) Quantization version backend. Default: `GEMM`.

### EXL2-Specific Parameters (used when `--method=exl2`)

* `--exl2_bits`: (Float) Target average bits per weight for the entire model. E.g., `4.0`, `6.5`, `8.0`. Default: `4.0`.
* `--exl2_head_bits`: (Integer) Target bits per weight for the head layer (typically higher to preserve output distribution). Default: `6`.

## Examples

**1. Basic AWQ Quantization**
```bash
uv run python quantize.py \
    --model_path unsloth/Meta-Llama-3.1-8B-Instruct \
    --calib_data datasets/my_calib.parquet \
    --quant_path output_awq_model \
    --method awq \
    --w_bit 4
```

**2. EXL2 Quantization with 6.5 bits**
```bash
uv run python quantize.py \
    --model_path unsloth/Meta-Llama-3.1-8B-Instruct \
    --calib_data datasets/my_calib.parquet \
    --quant_path output_exl2_model \
    --method exl2 \
    --exl2_bits 6.5 \
    --exl2_head_bits 8
```
