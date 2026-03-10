import urllib3

urllib3.disable_warnings()

import os
import time
import requests
import io
import numpy as np
import base64
from PIL import Image
from typing import Dict
import re
import hashlib
from glob import glob
from dotenv import load_dotenv
import json_repair

dotenv_path = os.path.join(os.path.dirname(__file__), "../.env")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path, verbose=True, override=True)
    print("Environment loaded")

def _array_to_jpeg_bytes(image: np.ndarray) -> bytes:
    """Converts a numpy array into a byte string for a JPEG image."""
    image = Image.fromarray(image)
    in_mem_file = io.BytesIO()
    image.save(in_mem_file, format="JPEG")
    in_mem_file.seek(0)
    img_bytes = in_mem_file.read()
    return img_bytes


def np_array_to_jpeg_base64(image: np.ndarray) -> str:
    """Encodes a numpy array image to JPEG base64 string."""
    return base64.b64encode(_array_to_jpeg_bytes(image)).decode("utf-8")

import base64

def str2base32(s: str) -> str:
    """
    将一个字符串编码为 Base32 格式的字符串。

    Args:
        s: 原始输入字符串。

    Returns:
        经过 Base32 编码后的字符串。
    """
    return base64.b32encode(s.encode('utf-8')).decode('ascii')

def base322str(s: str) -> str:
    """
    将一个 Base32 编码的字符串解码回原始字符串。

    Args:
        s: 经过 Base32 编码的字符串。

    Returns:
        解码后的原始字符串。
    """
    return base64.b32decode(s.encode('ascii')).decode('utf-8')


def openai_request(
    messages: list,
    model: str = "env",
    max_retry: int = 5,
    timeout: int = 60,
    temperature: float = 0.0,
    max_tokens: int = 300,
    usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0},
) -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f'Bearer {os.getenv("OPENAI_API_KEY")}',
    }
    data = {
        "model": os.getenv("OPENAI_API_MODEL", model) if model == "env" else model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    url = (
        f"{os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")}/chat/completions"
    )
    HTTP__PROXY = os.getenv("HTTP__PROXY", None)
    proxies = None
    if HTTP__PROXY:
        proxies = {
            "http": HTTP__PROXY,
            "https": HTTP__PROXY,
        }
    r = None
    for i in range(max_retry + 1):
        try:
            r = requests.post(
                url,
                headers=headers,
                json=data,
                timeout=timeout,
                verify=False,  # 禁用证书验证
                proxies=proxies,
            )  # .json()
            d = r.json()
            content = d.get("choices", [{}])[0].get("message", {})["content"]
            usage["prompt_tokens"] += d.get("usage", {}).get("prompt_tokens", 0)
            usage["completion_tokens"] += d.get("usage", {}).get("completion_tokens", 0)
            return content
        except Exception as e:
            print(
                f"Request failed: {e} , retrying {i+1} of {max_retry} after {(i + 1) ** 3} seconds"
            )
            if r is not None:
                print(r.text)
            time.sleep((i + 1) ** 3)
    raise Exception(f"Request failed after retrying {max_retry} times")

from .prompt_templates_web import GENERATE_INPUT_CONTENT_PROMPT

import numpy as np
from typing import Union
def generate_input_text_for_editable_element(
    marked_ndarray_screenshot: np.ndarray,
    ui_element_mark: Union[int, str],
    usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0},
) -> str:
    """
    Generate input text for editable elements.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": GENERATE_INPUT_CONTENT_PROMPT.format(index=ui_element_mark),
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{np_array_to_jpeg_base64(resize_ndarray_image(marked_ndarray_screenshot,target_max_size=1000))}",
                    },
                },
            ],
        }
    ]
    return openai_request(messages, max_tokens=300, usage=usage)


import pickle
import zstd


def save_object_to_disk(obj: object, file_path: str, compress_level: int = 3):
    """将对象序列化为pickle格式并使用Zstandard压缩保存到本地文件
    Args:
        obj (object): 要保存的对象
        file_path (str): 保存文件的路径
        compress_level (int): compression level, ultra-fast levels from -100 (ultra) to -1 (fast) available since zstd-1.3.4, and from 1 (fast) to 22 (slowest), 0 or unset - means default (3). Default 3.
    """
    pickled_data = pickle.dumps(obj)
    compressed_data = zstd.compress(pickled_data, compress_level)
    with open(file_path, "wb") as file:
        file.write(compressed_data)


def load_object_from_disk(file_path: str) -> object:
    """从本地文件读取Zstandard压缩的pickle数据并反序列化为对象"""
    with open(file_path, "rb") as file:
        compressed_data = file.read()
    pickled_data = zstd.decompress(compressed_data)
    return pickle.loads(pickled_data)


import ast
import re
import json
from typing import Any, Optional


def extract_json(s: str) -> Optional[dict[str, Any]]:
    """Extracts the first JSON object found in a string.

    Handles multi-line JSON and JSON embedded within other text.

    Args:
      s: A string potentially containing a JSON object.
         E.g., "{'hello': 'world'}" (Python-like) or '"key": "value", "boolean": true, "nothing": null' (Standard JSON) or CoT: "let's think step-by-step, ..., { ... json ... } ... more text"

    Returns:
      The parsed JSON object as a Python dictionary, or None if no valid
      JSON object is found or parsing fails.
    """
    pattern = r"\{.*\}"
    match = re.search(pattern, s, re.DOTALL)
    if match:
        potential_json_string = match.group()
        try:
            return json_repair.loads(potential_json_string)
        except Exception as json_error:
            print(
                f"JSON parsing failed ({json_error}), attempting Python literal eval."
            )
            try:
                return ast.literal_eval(potential_json_string)
            except (SyntaxError, ValueError) as eval_error:
                print(
                    f"Python literal eval also failed ({eval_error}), cannot extract dictionary."
                )
                return None
    else:
        return None


from PIL import Image
import numpy as np

def resize_pil_image(image: Image.Image, target_max_size: int = 1000) -> Image.Image:
    """
    Resize a PIL image to fit within a square of target_max_size x target_max_size pixels,
    maintaining the aspect ratio.
    """
    width, height = image.size
    if width > height:
        new_width = target_max_size
        new_height = int((height / width) * target_max_size)
    else:
        new_height = target_max_size
        new_width = int((width / height) * target_max_size)
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def resize_ndarray_image(image: np.ndarray, target_max_size: int = 1000) -> np.ndarray:
    """
    Resize a numpy ndarray image to fit within a square of target_max_size x target_max_size pixels, maintaining the aspect ratio.
    """
    return np.array(resize_pil_image(Image.fromarray(image), target_max_size))
import os
import json
from PIL import Image
from typing import Dict, Any
from dataclasses import asdict


def json_serializable(obj):
    if isinstance(obj, (np.ndarray, np.generic)):
        return obj.tolist()
    elif isinstance(obj, (Image.Image)):
        return np.array(obj).tolist()
    else:
        try:
            return asdict(obj)
        except Exception as e:
            pass
    raise TypeError(f"Type {type(obj)} not serializable")


import shutil
from .prompt_templates_web import TRAJECTORY_TO_INSTRUCTION_PROMPT

from .webarena_device import _generate_ui_element_description

def trajectory_to_instruction(
    trajectory_data: Dict[str, Any],
    usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0},
) -> None:
    """
    Convert the trajectory data to a high-level instruction and reasoning process and low-level instructions.
    """
    trajectory_information = ""
    for i in range(len(trajectory_data["actions"])):
        step_idx = i + 1  # 从1开始
        action = trajectory_data["actions"][i]["action_name"]
        action_type = trajectory_data["actions"][i]["action_type"]
        node = trajectory_data["actions"][i]["node"]
        current_pages_title = trajectory_data["current_pages_titles"][i]
        current_url = trajectory_data["current_urls"][i]
        trajectory_information += f"Step {step_idx}:\n"
        if action_type == "input":
            input_content = node.actions["input"]["input_content"]
            trajectory_information += f"Action: {action} `{input_content}` into {_generate_ui_element_description(node.ele,step_idx)}\n"
        else:
            trajectory_information += f"Action: {action} on {_generate_ui_element_description(node.ele,step_idx)}\n"
        trajectory_information += f"Current Page Title: {current_pages_title}\n"
        trajectory_information += f"Current URL: {current_url}\n\n"
    trajectory_information = trajectory_information.strip()  # 去掉最后的换行符

    text_prompt = TRAJECTORY_TO_INSTRUCTION_PROMPT.format(
        trajectory_information=(
            trajectory_information if trajectory_information else "None"
        )
    )
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text_prompt},
            ],
        }
    ]
    for screenshot_with_step in trajectory_data["screenshot_with_steps"]:
        messages[0]["content"].append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{np_array_to_jpeg_base64(resize_ndarray_image(screenshot_with_step, target_max_size=1000))}",
                },
            }
        )

    response_text = openai_request(
        messages=messages, timeout=600, max_tokens=16384, usage=usage
    )
    d = extract_json(response_text)
    assert d is not None, f"Failed to extract JSON from response: {response_text}"
    assert "Sub-Instruction" in d, f"Sub-Instruction not found in {response_text}"
    assert "Analysis" in d, f"Analysis not found in {response_text}"
    assert "Knowledge" in d, f"Knowledge not found in {response_text}"
    assert "Selected-Step-ID" in d, f"Selected-Step-ID not found in {response_text}"
    assert (
        "High-Level-Instruction" in d
    ), f"High-Level-Instruction not found in {response_text}"
    trajectory_data["low_level_instructions"] = d["Sub-Instruction"]
    trajectory_data["analyses"] = d["Analysis"]
    trajectory_data["knowledge"] = d["Knowledge"]
    trajectory_data["selected_step_idx"] = [i - 1 for i in d["Selected-Step-ID"]]
    trajectory_data["high_level_instruction"] = d["High-Level-Instruction"]
    assert len(trajectory_data["selected_step_idx"])>0, f"Selected-Step-ID is empty:\n{response_text}"

def get_md5_hash(input_str: str):
    hash_object = hashlib.md5(input_str.encode())
    return hash_object.hexdigest().upper()


def get_latest_filepath(folder: str, filename: str) -> str:
    """
    Construct a new file path by combining the folder and filename like this:
    folder/001_filename
    """

    def extract_cnt(filepath: str) -> int:
        """
        Extract the count from the filename like this:
        folder/001_filename -> 1
        """
        return int(os.path.basename(filepath).split("_")[0])

    fps = glob(os.path.join(folder, f"*_{filename}"))
    if not fps:
        return None
    fps = sorted(fps, key=lambda x: extract_cnt(x))
    return fps[-1]


def construct_new_filepath(folder: str, filename: str) -> str:
    """
    Construct a new file path by combining the folder and filename like this:
    folder/001_filename
    """

    def extract_cnt(filepath: str) -> int:
        """
        Extract the count from the filename like this:
        folder/001_filename -> 1
        """
        return int(os.path.basename(filepath).split("_")[0])

    fps = glob(os.path.join(folder, f"*_{filename}"))
    if not fps:
        return os.path.join(folder, f"001_{filename}")
    fps = sorted(fps, key=lambda x: extract_cnt(x))
    cnt = extract_cnt(fps[-1])
    assert (
        cnt < 999
    ), f"cnt is out of range: {cnt}, please delete some files in {folder}"
    cnt += 1
    return os.path.join(folder, f"{cnt:03d}_{filename}")

def update_documents(
    trajectory_data: Dict[str, Any], documents: Dict[str, str]
) -> None:
    """
    更新文档数据
    """
    for i in range(len(trajectory_data["actions"])):
        action = trajectory_data["actions"][i]
        node = action["node"]
        if node.uid not in documents and i < len(trajectory_data["knowledge"]):
            documents[node.uid] = trajectory_data["knowledge"][i]
