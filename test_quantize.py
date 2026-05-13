import json
import pandas as pd
import pytest
from transformers import AutoTokenizer
from quantize import load_and_format_calibration_data

@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("t-tech/T-lite-it-2.1", trust_remote_code=True)

def test_load_and_format_calibration_data(tmp_path, tokenizer):
    # Create mock data
    mock_data = [
        {
            "request": json.dumps({
                "messages": [{"role": "user", "content": "What is 2+2?"}],
                "tools": [{"type": "function", "function": {"name": "calc", "description": "calculator"}}]
            }),
            "response": json.dumps({
                "choices": [{"message": {"role": "assistant", "content": "4", "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "calc", "arguments": "{\"expr\": \"2+2\"}"}}]}}]
            })
        },
        {
            "request": json.dumps({
                "messages": [{"role": "user", "content": "Output in JSON."}],
                "response_format": {"type": "json_object"}
            }),
            "response": json.dumps({
                "choices": [{"message": {"role": "assistant", "content": "{\"status\": \"ok\"}"}}]
            })
        },
        {
            "request": json.dumps({
                "messages": [{"role": "user", "content": "Just a normal message"}]
            }),
            "response": None
        }
    ]
    df = pd.DataFrame(mock_data)
    parquet_path = tmp_path / "mock_calib.parquet"
    df.to_parquet(parquet_path)

    formatted_data = load_and_format_calibration_data(str(parquet_path), tokenizer)

    assert len(formatted_data) == 3

    # Check that tools prompt is injected
    assert "<tools>" in formatted_data[0]
    assert '{"name": "calc", "arguments": {"expr": "2+2"}}' in formatted_data[0]

    # Check response format injection
    # In some tokenizers, response_format may not add specific text, but we should verify it doesn't crash
    assert "{\"status\": \"ok\"}" in formatted_data[1]

    assert "Just a normal message" in formatted_data[2]
