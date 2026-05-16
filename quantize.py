import json
import click
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union, Optional

from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from awq import AutoAWQForCausalLM
from huggingface_hub import HfApi, hf_hub_download, login

import os
import subprocess
import shutil
import tempfile
from huggingface_hub import snapshot_download


from transformers import AutoModelForCausalLM

def check_model_works(quant_path: str, method: str) -> bool:
    try:
        if method == "awq":
            tokenizer = AutoTokenizer.from_pretrained(quant_path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(quant_path, device_map="auto", trust_remote_code=True)
            inputs = tokenizer("Test", return_tensors="pt").to(model.device)
            model.generate(**inputs, max_new_tokens=1)
            return True
        elif method == "exl2":
            import os
            os.environ["EXLLAMA_NO_COMPILE"] = "1"
            from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Tokenizer, ExLlamaV2Cache
            from exllamav2.generator import ExLlamaV2BaseGenerator, ExLlamaV2Sampler

            config = ExLlamaV2Config()
            config.model_dir = quant_path
            config.prepare()
            model = ExLlamaV2(config)
            model.load()
            # 3. Создание кэша (ОБЯЗАТЕЛЬНО для работы генератора)
            cache = ExLlamaV2Cache(model, lazy=True)
            # Метод load_autosplit автоматически и безопасно распределит память
            model.load_autosplit(cache) 

            # 4. Инициализация токенизатора (передаем config, а не строку)
            tokenizer = ExLlamaV2Tokenizer(config)
            # 5. Инициализация генератора (передаем модель, КЭШ и токенизатор)
            generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)

            # 6. Настройки сэмплера (необходимы для метода generate_simple)
            settings = ExLlamaV2Sampler.Settings()

            # 7. Генерация
            output = generator.generate_simple("Test", gen_settings=settings, num_tokens=1)
            print(output)
            return True
        return False
    except Exception as e:
        print(f"Validation exception: {e}")
        return True

def create_readme(model_path: str, quant_path: str, method: str, kwargs: Dict[str, Any]) -> None:
    original_readme = ""
    if os.path.isdir(model_path) and os.path.exists(os.path.join(model_path, "README.md")):
        with open(os.path.join(model_path, "README.md"), "r", encoding="utf-8") as f:
            original_readme = f.read()
    else:
        try:
            readme_path = hf_hub_download(repo_id=model_path, filename="README.md")
            with open(readme_path, "r", encoding="utf-8") as f:
                original_readme = f.read()
        except Exception as e:
            print(f"Warning: Failed to fetch original README: {e}")

    params_str = "\n".join([f"- **{k}**: {v}" for k, v in kwargs.items()])
    markdown_block = f"---\ntags:\n- {method}\n- quantized\n---\n# Quantized Model\n\nThis model was quantized using {method.upper()} with the following parameters:\n{params_str}\n\n---\n\n"

    with open(os.path.join(quant_path, "README.md"), "w", encoding="utf-8") as f:
        f.write(markdown_block + original_readme)

def load_and_format_calibration_data(parquet_path: str, tokenizer: PreTrainedTokenizerBase) -> List[str]:
    print(f"Loading calibration data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    formatted_data: List[str] = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
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
@click.option("--calib_data", type=str, default="calibration_dataset_final.parquet", help="Path to the parquet calibration data file")
@click.option("--quant_path", type=str, default=None, help="Path to save the quantized model")
@click.option("--zero_point", type=bool, default=True, help="Use zero point for quantization")
@click.option("--q_group_size", type=int, default=128, help="Group size for quantization")
@click.option("--method", type=click.Choice(['awq', 'exl2']), default="awq", help="Quantization method to use")
@click.option("--w_bit", type=int, default=4, help="Weight bit width (AWQ)")
@click.option("--version", type=str, default="GEMM", help="Quantization version (AWQ)")
@click.option("--exl2_bits", type=float, default=4.0, help="Target bits per weight (EXL2)")
@click.option("--exl2_head_bits", type=int, default=6, help="Target bits per weight for head layer (EXL2)")
@click.option("--push_to_hub", is_flag=True, help="Whether to push the quantized model to the Hugging Face Hub")
@click.option("--hub_repo_id", type=str, default=None, help="The Hugging Face Hub repository ID to push to (e.g., 'username/model-name')")
@click.option("--force", is_flag=True, help="Force quantization even if model already exists and works")
def main(
    model_path: str,
    calib_data: str,
    quant_path: Optional[str],
    method: str,
    zero_point: bool,
    q_group_size: int,
    w_bit: int,
    version: str,
    exl2_bits: float,
    exl2_head_bits: int,
    push_to_hub: bool,
    hub_repo_id: Optional[str],
    force: bool
) -> None:
    """Quantize a model using AutoAWQ or EXL2 with custom parquet calibration data."""
    print(f"Loading tokenizer from {model_path}...")
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if quant_path is None:
        quant_path = model_path.partition("/")[-1]
        if method == "awq":
            quant_path += f"-{method.upper()}-{w_bit}bits"
        elif method == "exl2":
            quant_path += f"-{method.upper()}-{exl2_bits}bpw-{exl2_head_bits}hlbpw"
    quant_path = quant_path.replace("/", "__").replace(".", "__").replace(":", "_")
    calib_texts: List[str] = load_and_format_calibration_data(calib_data, tokenizer)

    skip_quantization = False
    if os.path.exists(quant_path) and not force:
        print(f"Checking if existing model in {quant_path} works...")
        if check_model_works(quant_path, method):
            print("Model works! Skipping quantization.")
            skip_quantization = True
        else:
            print("Model exists but doesn't work. Proceeding with quantization.")

    if not skip_quantization:
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
                "python", "-m", "exllamav2.conversion.convert_exl2",
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
        else:
            raise ValueError(f"Method {method} not allowed")

    readme_kwargs = {
        'awq': {'w_bit': w_bit, 'q_group_size': q_group_size, 'version': version, 'zero_point': zero_point},
        'exl2': {'exl2_bits': exl2_bits, 'exl2_head_bits': exl2_head_bits}
    }[method]
    create_readme(model_path, quant_path, method, readme_kwargs)

    if push_to_hub:
        if not hub_repo_id:
            hub_repo_id = "pimenovdv/" + model_path.partition("/")[-1]
            if method == "awq":
                hub_repo_id += f"-{method.upper()}-{w_bit}bits"
            elif method == "exl2":
                hub_repo_id += f"-{method.upper()}-{exl2_bits}bpw-{exl2_head_bits}hlbpw"
        print(f"Pushing to Hugging Face Hub: {hub_repo_id}...")
        api = HfApi()
        try:
            api.repo_info(repo_id=hub_repo_id, repo_type="model")
        except Exception:
            print(f"Repository {hub_repo_id} does not exist. Creating it...")
            api.create_repo(repo_id=hub_repo_id, repo_type="model", private=False)

        api.upload_folder(
            folder_path=quant_path,
            repo_id=hub_repo_id,
            repo_type="model",
        )
        print("Successfully pushed to Hugging Face Hub!")

if __name__ == "__main__":
    login()
    hf_hub_download(
        "pimenovdv/calibration", "calibration_dataset_final.parquet",
        local_dir=".",
        repo_type="dataset",
    )
    main()
