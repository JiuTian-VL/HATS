import dataclasses
from typing import Any, Optional
import xml.etree.ElementTree as ET
from dataclasses import asdict
import time
from PIL import Image
import imagehash
import numpy as np

import ast
import re
import json
import requests
import os
import io
import base64

from PIL import Image
import numpy as np
import pickle
import zstd
import json_repair


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
    return image.resize((new_width, new_height), Image.LANCZOS)


def resize_pil_image_qwen(
    image: Image.Image, target_max_size: int = 1000
) -> Image.Image:
    """
    Resize a PIL image to fit within a square of target_max_size x target_max_size pixels,
    maintaining the aspect ratio.
    """
    return image.resize((target_max_size, target_max_size), Image.LANCZOS)


def resize_ndarray_image_qwen(
    image: np.ndarray, target_max_size: int = 1000
) -> np.ndarray:
    """
    Resize a numpy ndarray image to fit within a square of target_max_size x target_max_size pixels, maintaining the aspect ratio.
    """
    return np.array(resize_pil_image_qwen(Image.fromarray(image), target_max_size))


def resize_ndarray_image(image: np.ndarray, target_max_size: int = 1000) -> np.ndarray:
    """
    Resize a numpy ndarray image to fit within a square of target_max_size x target_max_size pixels, maintaining the aspect ratio.
    """
    return np.array(resize_pil_image(Image.fromarray(image), target_max_size))


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


def pil_to_webp_base64(img: Image.Image) -> str:
    buffered = io.BytesIO()
    img.convert("RGB").save(buffered, format="WEBP", quality=95)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def ndarray_to_webp_base64(img: np.ndarray) -> str:
    """
    Convert a numpy ndarray image to a base64 encoded string.
    """
    return pil_to_webp_base64(Image.fromarray(img))


def openai_request(
    messages: list,
    model: str,
    openai_api_key: str,
    openai_base_url: str = "https://api.openai.com/v1",
    max_retry: int = 5,
    timeout: int = 60,
    temperature: float = 0.0,
    max_tokens: int = 300,
    usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0},
) -> str:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}",
    }
    data = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    url = f"{openai_base_url}/chat/completions"
    proxies = None
    r = None
    for i in range(max_retry + 1):
        try:
            r = requests.post(
                url,
                headers=headers,
                json=data,
                timeout=timeout,
                proxies=proxies,
            )
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


@dataclasses.dataclass
class BoundingBox:
    """Class for representing a bounding box."""

    x_min: float | int
    x_max: float | int
    y_min: float | int
    y_max: float | int

    @property
    def center(self) -> tuple[float, float]:
        """Gets center of bounding box."""
        return (self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0

    @property
    def width(self) -> float | int:
        """Gets width of bounding box."""
        return self.x_max - self.x_min

    @property
    def height(self) -> float | int:
        """Gets height of bounding box."""
        return self.y_max - self.y_min

    @property
    def area(self) -> float | int:
        return self.width * self.height


@dataclasses.dataclass
class UIElement:
    """Represents a UI element."""

    text: Optional[str] = None
    content_description: Optional[str] = None
    class_name: Optional[str] = None
    bbox: Optional[BoundingBox] = None
    bbox_pixels: Optional[BoundingBox] = None
    hint_text: Optional[str] = None
    is_checked: Optional[bool] = None
    is_checkable: Optional[bool] = None
    is_clickable: Optional[bool] = None
    is_editable: Optional[bool] = None
    is_enabled: Optional[bool] = None
    is_focused: Optional[bool] = None
    is_focusable: Optional[bool] = None
    is_long_clickable: Optional[bool] = None
    is_scrollable: Optional[bool] = None
    is_selected: Optional[bool] = None
    is_visible: Optional[bool] = None
    package_name: Optional[str] = None
    resource_name: Optional[str] = None
    tooltip: Optional[str] = None
    resource_id: Optional[str] = None

    element_id: Optional[str] = None
    _image_hash: Optional[str] = dataclasses.field(
        default=None, repr=False
    )
    uid: Optional[str] = None
    hints: Optional[list[str]] = dataclasses.field(default_factory=list, repr=False)

    def _get_element_id(self) -> str:
        """Generates a unique ID for the UI element based on its properties."""
        if self.element_id is None and self.bbox_pixels:
            elem_w, elem_h = self.bbox_pixels.width, self.bbox_pixels.height
            if self.resource_id:
                self.element_id = self.resource_id.replace(":", ".").replace("/", "_")
            else:
                self.element_id = f"{self.class_name}_{elem_w}_{elem_h}"
            if self.content_description and len(self.content_description) < 20:
                content_desc = (
                    self.content_description.replace("/", "_")
                    .replace(" ", "")
                    .replace(":", "_")
                )
                self.element_id += f"_{content_desc}"
        return self.element_id

    def _maybe_is_editable(self) -> bool:
        """Checks if the UI element is editable based on its properties."""
        if self.is_editable:
            return True
        if self.is_clickable and self.class_name:
            for s in [
                "EditText",
                "TextView",
                "AutoCompleteTextView",
                "MultiAutoCompleteTextView",
            ]:
                if s in self.class_name:
                    return True
        return False

    def _update_uid(self) -> None:
        if self.element_id is not None and self._image_hash is not None:
            self.uid = f"{self.element_id}_{self._image_hash }"
        elif self._image_hash is not None:
            self.uid = f"{self._image_hash }"
        elif self.element_id is not None:
            self.uid = f"{self.element_id}"

    # Define the property getter and setter
    @property
    def image_hash(self) -> Optional[str]:
        """Getter for the image hash."""
        return self._image_hash

    @image_hash.setter
    def image_hash(self, value: Optional[str]) -> None:
        """Setter for the image hash that also updates the UID."""
        # Optional: Add validation for 'value' if needed
        if self._image_hash != value:  # Optional: Avoid update if value hasn't changed
            self._image_hash = value
            self._update_uid()  # Call uid update *after* setting the new value

    def __post_init__(self):
        """Post-initialization to ensure element_id is set."""
        if self.bbox_pixels is not None and isinstance(self.bbox_pixels, dict):
            self.bbox_pixels = BoundingBox(**self.bbox_pixels)
        if self.bbox is not None and isinstance(self.bbox, dict):
            self.bbox = BoundingBox(**self.bbox)
        self.element_id = self._get_element_id()
        self.is_editable = self._maybe_is_editable()
        self._update_uid()


def _normalize_bounding_box(
    node_bbox: BoundingBox,
    screen_width_height_px: tuple[int, int],
) -> BoundingBox:
    width, height = screen_width_height_px
    return BoundingBox(
        node_bbox.x_min / width,
        node_bbox.x_max / width,
        node_bbox.y_min / height,
        node_bbox.y_max / height,
    )


def _parse_ui_hierarchy(xml_string: str) -> dict[str, Any]:
    """Parses the UI hierarchy XML into a dictionary structure."""
    root = ET.fromstring(xml_string)

    def parse_node(node):
        result = node.attrib
        result["children"] = [parse_node(child) for child in node]
        return result

    return parse_node(root)


def xml_dump_to_ui_elements(
    xml_string: str,
    exclude_invisible_elements: bool = False,
    screen_size: Optional[tuple[int, int]] = None,
    screenshot: Optional[Image.Image] = None,
) -> list[UIElement]:
    """Converts a UI hierarchy XML dump from uiautomator dump to UIElements.
    Args:
        xml_string: The XML string containing the UI hierarchy dump.
        exclude_invisible_elements: True if invisible elements should not be
      returned.
        screen_size: The size of the device screen in pixels (width, height).

    Returns:
        The extracted UI elements.
    """

    def text_or_none(text: Optional[str]) -> Optional[str]:
        """Returns None if text is None or 0 length."""
        return text if text else None

    parsed_hierarchy = _parse_ui_hierarchy(xml_string)
    ui_elements = []

    def process_node(node, screen_size=None, is_root=False, parent_node=None):
        bounds = node.get("bounds")
        bbox_pixels, bbox_normalized = None, None
        if bounds:
            x_min, y_min, x_max, y_max = map(
                int, bounds.strip("[]").replace("][", ",").split(",")
            )
            bbox_pixels = BoundingBox(x_min, x_max, y_min, y_max)
            if screen_size is not None:
                bbox_normalized = _normalize_bounding_box(bbox_pixels, screen_size)

        ui_element = UIElement(
            text=text_or_none(node.get("text")),
            content_description=text_or_none(node.get("content-desc")),
            class_name=text_or_none(node.get("class")),
            bbox=bbox_normalized,
            bbox_pixels=bbox_pixels,
            hint_text=text_or_none(node.get("hint")),
            is_checked=node.get("checked") == "true",
            is_checkable=node.get("checkable") == "true",
            is_clickable=node.get("clickable") == "true",
            is_enabled=node.get("enabled") == "true",
            is_focused=node.get("focused") == "true",
            is_focusable=node.get("focusable") == "true",
            is_long_clickable=node.get("long-clickable") == "true",
            is_scrollable=node.get("scrollable") == "true",
            is_selected=node.get("selected") == "true",
            package_name=text_or_none(node.get("package")),
            resource_id=text_or_none(node.get("resource-id")),
            is_visible=node.get("visible-to-user") == "true",
        )
        if parent_node and parent_node.element_id:
            pass
        if not is_root:
            if (
                not (node.get("children", None) is not None)
                or (text_or_none(node.get("content-desc")) is not None)
                or (node.get("scrollable", "false") == "true")
                or (node.get("clickable", "false") == "true")
            ):
                if exclude_invisible_elements and not (
                    node.get("visible-to-user", "false") == "true"
                ):
                    pass
                else:
                    if screen_size is None or validate_ui_element(
                        ui_element, screen_size
                    ):
                        if ui_element.bbox_pixels and screenshot:
                            image_hash = imagehash.phash(
                                screenshot.crop(
                                    (
                                        ui_element.bbox_pixels.x_min,
                                        ui_element.bbox_pixels.y_min,
                                        ui_element.bbox_pixels.x_max,
                                        ui_element.bbox_pixels.y_max,
                                    )
                                ),
                                hash_size=16,
                                highfreq_factor=8,
                            )
                            ui_element.image_hash = str(image_hash).upper()
                        ui_elements.append(ui_element)

        for child in node.get("children", []):
            process_node(
                child, screen_size=screen_size, is_root=False, parent_node=ui_element
            )

    process_node(parsed_hierarchy, screen_size=screen_size, is_root=True)
    return ui_elements


def _covert_bool_ndarray_to_01str(bool_ndarray: np.ndarray) -> str:
    """Convert a boolean ndarray to a string of 0s and 1s.
    Args:
        bool_ndarray (np.ndarray): A boolean ndarray.
    Returns:
        str: A string representation of the boolean ndarray, where True is represented as '1' and False as '0'.
    """
    assert bool_ndarray.dtype == np.bool_, "Input array must be of type np.bool_"
    return "".join("1" if x else "0" for x in bool_ndarray.flatten())


def _generate_ui_element_description(ui_element: UIElement, index: int) -> str:
    """Generate a description for a given UI element with important information.

    Args:
      ui_element: UI elements for the current screen.
      index: The numeric index for the UI element.

    Returns:
      The description for the UI element.
    """
    element_description = f'UI element {index}: {{"index": {index}, '
    if ui_element.text:
        element_description += f'"text": "{ui_element.text}", '
    if ui_element.content_description:
        element_description += (
            f'"content_description": "{ui_element.content_description}", '
        )
    if ui_element.hint_text:
        element_description += f'"hint_text": "{ui_element.hint_text}", '
    if ui_element.tooltip:
        element_description += f'"tooltip": "{ui_element.tooltip}", '
    element_description += (
        f'"is_clickable": {"True" if ui_element.is_clickable else "False"}, '
    )
    element_description += (
        '"is_long_clickable":'
        f' {"True" if ui_element.is_long_clickable else "False"}, '
    )
    element_description += (
        f'"is_editable": {"True" if ui_element.is_editable else "False"}, '
    )
    if ui_element.is_scrollable:
        element_description += '"is_scrollable": True, '
    if ui_element.is_focusable:
        element_description += '"is_focusable": True, '
    element_description += (
        f'"is_selected": {"True" if ui_element.is_selected else "False"}, '
    )
    element_description += (
        f'"is_checked": {"True" if ui_element.is_checked else "False"}, '
    )
    return element_description[:-2] + "}"  # 这里的[:-2]是为了去掉', '


def validate_ui_element(
    ui_element: UIElement,
    screen_width_height_px: tuple[int, int],
) -> bool:
    """Used to filter out invalid UI element."""
    screen_width, screen_height = screen_width_height_px

    # Filters out invisible element.
    if not ui_element.is_visible:
        return False

    # Filters out element with invalid bounding box.
    if ui_element.bbox_pixels:
        x_min = ui_element.bbox_pixels.x_min
        x_max = ui_element.bbox_pixels.x_max
        y_min = ui_element.bbox_pixels.y_min
        y_max = ui_element.bbox_pixels.y_max

        if (
            x_min >= x_max
            or x_min >= screen_width
            or x_max <= 0
            or y_min >= y_max
            or y_min >= screen_height
            or y_max <= 0
        ):
            return False

    return True


def _generate_ui_elements_description_list(
    ui_elements: list[UIElement],
    screen_width_height_px: tuple[int, int],
) -> str:
    """Generate concise information for a list of UIElement.

    Args:
      ui_elements: UI elements for the current screen.
      screen_width_height_px: The height and width of the screen in pixels.

    Returns:
      Concise information for each UIElement.
    """
    tree_info = ""
    for index, ui_element in enumerate(ui_elements):
        if validate_ui_element(ui_element, screen_width_height_px):
            tree_info += _generate_ui_element_description(ui_element, index) + "\n"
    return tree_info


import base64
import re
from typing import Any, Optional
import cv2
import numpy as np


def _logical_to_physical(
    logical_coordinates: tuple[int, int],
    logical_screen_size: tuple[int, int],
    physical_frame_boundary: tuple[int, int, int, int],
    orientation: int,
) -> tuple[int, int]:
    """Convert logical coordinates to physical coordinates.

    Args:
      logical_coordinates: The logical coordinates for the point.
      logical_screen_size: The logical screen size.
      physical_frame_boundary: The physical coordinates in portrait orientation
        for the upper left and lower right corner for the frame.
      orientation: The current screen orientation.

    Returns:
      The physical coordinate for the point in portrait orientation.

    Raises:
      ValueError: If the orientation is not valid.
    """
    x, y = logical_coordinates
    px0, py0, px1, py1 = physical_frame_boundary
    px, py = px1 - px0, py1 - py0
    lx, ly = logical_screen_size
    if orientation == 0:
        return (int(x * px / lx) + px0, int(y * py / ly) + py0)
    if orientation == 1:
        return (px - int(y * px / ly) + px0, int(x * py / lx) + py0)
    if orientation == 2:
        return (px - int(x * px / lx) + px0, py - int(y * py / ly) + py0)
    if orientation == 3:
        return (int(y * px / ly) + px0, py - int(x * py / lx) + py0)
    print("Invalid orientation.")
    raise ValueError("Unsupported orientation.")


def _ui_element_logical_corner(
    ui_element: UIElement, orientation: int
) -> list[tuple[int, int]]:
    """Get logical coordinates for corners of a given UI element.

    Args:
      ui_element: The corresponding UI element.
      orientation: The current orientation.

    Returns:
      Logical coordinates for upper left and lower right corner for the UI
      element.

    Raises:
      ValueError: If bounding box is missing.
      ValueError: If orientation is not valid.
    """
    if ui_element.bbox_pixels is None:
        raise ValueError("UI element does not have bounding box.")
    if orientation == 0:
        return [
            (int(ui_element.bbox_pixels.x_min), int(ui_element.bbox_pixels.y_min)),
            (int(ui_element.bbox_pixels.x_max), int(ui_element.bbox_pixels.y_max)),
        ]
    if orientation == 1:
        return [
            (int(ui_element.bbox_pixels.x_min), int(ui_element.bbox_pixels.y_max)),
            (int(ui_element.bbox_pixels.x_max), int(ui_element.bbox_pixels.y_min)),
        ]
    if orientation == 2:
        return [
            (int(ui_element.bbox_pixels.x_max), int(ui_element.bbox_pixels.y_max)),
            (int(ui_element.bbox_pixels.x_min), int(ui_element.bbox_pixels.y_min)),
        ]
    if orientation == 3:
        return [
            (int(ui_element.bbox_pixels.x_max), int(ui_element.bbox_pixels.y_min)),
            (int(ui_element.bbox_pixels.x_min), int(ui_element.bbox_pixels.y_max)),
        ]
    raise ValueError("Unsupported orientation.")


def add_ui_element_mark(
    screenshot: np.ndarray,
    ui_element: UIElement,
    index: int | str,
    logical_screen_size: tuple[int, int],
    physical_frame_boundary: tuple[int, int, int, int],
    orientation: int,
):
    """Add mark (a bounding box plus index) for a UI element in the screenshot.

    Args:
      screenshot: The screenshot as a numpy ndarray.
      ui_element: The UI element to be marked.
      index: The index for the UI element.
      logical_screen_size: The logical screen size.
      physical_frame_boundary: The physical coordinates in portrait orientation
        for the upper left and lower right corner for the frame.
      orientation: The current screen orientation.
    """
    if ui_element.bbox_pixels:
        upper_left_logical, lower_right_logical = _ui_element_logical_corner(
            ui_element, orientation
        )
        upper_left_physical = _logical_to_physical(
            upper_left_logical,
            logical_screen_size,
            physical_frame_boundary,
            orientation,
        )
        lower_right_physical = _logical_to_physical(
            lower_right_logical,
            logical_screen_size,
            physical_frame_boundary,
            orientation,
        )

        cv2.rectangle(
            screenshot,
            upper_left_physical,
            lower_right_physical,
            color=(0, 255, 0),
            thickness=3,
        )
        screenshot[
            upper_left_physical[1] + 1 : upper_left_physical[1] + 25,
            upper_left_physical[0] + 1 : upper_left_physical[0] + 35,
            :,
        ] = (255, 255, 255)
        cv2.putText(
            screenshot,
            str(index),
            (
                upper_left_physical[0] + 1,
                upper_left_physical[1] + 20,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            thickness=2,
        )


def add_screenshot_label(screenshot: np.ndarray, label: str):
    """Add a text label to the right bottom of the screenshot.

    Args:
      screenshot: The screenshot as a numpy ndarray.
      label: The text label to add, just a single word.
    """
    if len(label) > 8:
        print(f"Label {label} is too long, please use a shorter one.")
    height, width, _ = screenshot.shape
    screenshot[height - 30 : height, width - 150 : width, :] = (255, 255, 255)
    cv2.putText(
        screenshot,
        label,
        (width - 135, height - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        thickness=3,
    )


def parse_reason_action_output(
    raw_reason_action_output: str,
) -> tuple[Optional[str], Optional[str]]:
    r"""Parses llm action reason output.

    Args:
      raw_reason_action_output: Raw string output that supposes to have the format
        'Reasoning: xxx\nAction:xxx'.

    Returns:
      If parsing successfully, returns reason and action.
    """
    reason_result = re.search(
        r"Reasoning:(.*)Action:", raw_reason_action_output, flags=re.DOTALL
    )
    reason = reason_result.group(1).strip() if reason_result else None
    action_result = re.search(r"Action:(.*)", raw_reason_action_output, flags=re.DOTALL)
    action = action_result.group(1).strip() if action_result else None
    return reason, action


import logging
import uiautomator2 as u2
from PIL import Image
from typing import List, Tuple, Union
import time
import re


def get_available_devices() -> list[str]:
    """
    Get a list of device serials connected via adb
    :return: list of str, each str is a device serial number
    """
    import subprocess

    r = subprocess.check_output(["adb", "devices"])
    if not isinstance(r, str):
        r = r.decode()
    devices = []
    for line in r.splitlines():
        segs = line.strip().split()
        if len(segs) == 2 and segs[1] == "device":
            devices.append(segs[0])
    return devices


class Device(object):

    def __init__(self, device_serial: str = None) -> None:
        """
        Initialize a device connection with the bare minimum requirements.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        if device_serial is None:
            all_devices = get_available_devices()
            if len(all_devices) == 0:
                raise Exception("No device connected.")
            device_serial = all_devices[0]
        self.logger.info(f"Using device {device_serial}")
        self.device_serial = device_serial
        self.u2d = None
        self.connect()

    def __del__(self) -> None:
        self.disconnect()

    def connect(self) -> None:
        """
        Connect to the device.
        """
        self._prior_ui_elements_state = None
        if self.u2d is None:
            self.u2d = u2.connect(self.device_serial)
        self.logger.info(f"Connected to device.\n{self.u2d.info}")

    def disconnect(self) -> None:
        """
        Disconnect from the device.
        """
        if self.u2d is not None:
            self.u2d.stop_uiautomator()
            self.u2d = None
        self.logger.info("Disconnected from device.")

    def run_shell_command(
        self, cmdargs: Union[str, List[str]], timeout=60
    ) -> tuple[str, int]:
        """
        Run shell command on device

        Args:
            cmdargs: str or list, example: "ls -l" or ["ls", "-l"]
            timeout: seconds of command run, works on when stream is False

        Returns:
            return type is `namedtuple("ShellResponse", ("output", "exit_code"))`

        Raises:
            AdbShellError
        """
        return self.u2d.shell(cmdargs, timeout=timeout)

    def launch_app(
        self,
        package_name: str,
        use_monkey: bool = False,
        timeout: float = 20.0,
        front: bool = True,
        activity: str = None,
    ) -> None:
        """
        Args:
            package_name (str): package name
            use_monkey (bool): use monkey command to start app when activity is not given
            timeout (float): maxium wait time, 0 means no wait
            front (bool): wait until app is current app
        """
        self.u2d.app_start(package_name, use_monkey=use_monkey, activity=activity)
        if timeout > 0:
            self.u2d.app_wait(package_name, front=front, timeout=timeout)

    def stop_app(self, package_name: str):
        """Stop one application"""
        self.u2d.app_stop(package_name)

    def stop_all_apps(self, excludes: list = []) -> List[str]:
        """Stop all third party applications
        Args:
            excludes (list): apps that do not want to kill

        Returns:
            a list of killed apps
        """
        return self.u2d.app_stop_all(excludes=excludes)

    def list_running_app(self) -> List[str]:
        """
        Returns:
            list of running apps
        """
        return self.u2d.app_list_running()

    def list_installed_app(self, filter: str = None) -> List[str]:
        """
        List installed app package names

        Args:
            filter: [-f] [-d] [-e] [-s] [-3] [-i] [-u] [--user USER_ID] [FILTER]

        Returns:
            list of apps by filter
        """
        return self.u2d.app_list(filter)

    def get_viewhierachy(self) -> str:
        viewhierachy = self.u2d.dump_hierarchy(
            compressed=False, pretty=False, max_depth=50
        )
        return viewhierachy

    def get_screenshot(self) -> Image.Image:
        return self.u2d.screenshot().convert("RGB")

    def get_screen_size(self) -> Tuple[int, int]:
        """
        Returns:
            screen width and height
        """
        return self.u2d.window_size()

    def get_top_activity_name(self) -> str:
        current = self.u2d.app_current()
        return current["activity"]

    def get_top_package_name(self) -> str:
        current = self.u2d.app_current()
        return current["package"]

    def get_installed_apps(self) -> List[str]:
        return self.u2d.app_list()

    def click(self, x: int, y: int):
        self.u2d.click(x, y)

    def long_click(self, x: int, y: int, duration: float = 2.0):
        self.u2d.long_click(x, y, duration)

    def double_click(self, x: int, y: int):
        self.u2d.double_click(x, y)

    def drag(self, x1: int, y1: int, x2: int, y2: int, duration: float = 0.5):
        self.u2d.drag(x1, y1, x2, y2, duration)

    def swipe(self, x1: int, y1: int, x2: int, y2: int, duration: float = 0.5):
        self.u2d.swipe(x1, y1, x2, y2, duration)

    def swip_up(self):
        self.u2d.swipe_ext("up")

    def swip_down(self):
        self.u2d.swipe_ext("down")

    def swip_left(self):
        self.u2d.swipe_ext("left")

    def swip_right(self):
        self.u2d.swipe_ext("right")

    def is_keyboard_shown(self) -> bool:
        output, exit_code = self.u2d.shell(
            "dumpsys input_method | grep mInputShown", timeout=120
        )
        return "mInputShown=true" in output

    def input_text(self, text: str, smart_enter=True, clear_first=True):
        self.u2d.send_keys(
            text, clear=clear_first
        )
        if smart_enter:
            self.u2d.send_action()

    def enter(self):
        self.u2d.press("enter")

    def home(self):
        self.u2d.press("home")

    def back(self):
        self.u2d.press("back")

    def _get_ui_elements(
        self, exclude_invisible_elements: bool = True
    ) -> list[UIElement]:
        """Get the current UI elements from the device.

        Args:
            exclude_invisible_elements: If True, invisible elements will be excluded from the result.
        Returns:
            list[UIElement] - The extracted UI elements.
        """
        return xml_dump_to_ui_elements(
            self.get_viewhierachy(),
            exclude_invisible_elements=exclude_invisible_elements,
            screen_size=self.get_screen_size(),
            screenshot=self.get_screenshot(),
        )

    def wait_to_stabilize(
        self,
        stability_threshold: int = 3,
        sleep_duration: float = 0.5,
        timeout: float = 6.0,
    ) -> list[UIElement]:
        """Checks if the UI elements remain stable over a number of checks and returns the state.

        Args:
            stability_threshold: Number of consecutive checks where UI elements must
            remain the same to consider UI stable.
            sleep_duration: Minimum time in seconds between each check.
            timeout: Maximum time in seconds to wait for UI to become stable before
            giving up.

        Returns:
            The current state of the UI if stability is achieved within the timeout.
        """
        if not self._prior_ui_elements_state:
            self._prior_ui_elements_state = self._get_ui_elements()
        if stability_threshold <= 0:
            raise ValueError("Stability threshold must be a positive integer.")

        stable_checks = 1
        start_time = time.time()
        deadline = start_time + timeout
        current_ui_elements_state = []
        while stable_checks < stability_threshold and time.time() < deadline:
            iteration_start_time = time.time()
            current_ui_elements_state = self._get_ui_elements()

            if self._prior_ui_elements_state == current_ui_elements_state:
                stable_checks += 1
                if stable_checks == stability_threshold:
                    break  # Exit early if stability is achieved.
            else:
                stable_checks = 1  # Reset if any change is detected
                self._prior_ui_elements_state = current_ui_elements_state

            elapsed_time = time.time() - iteration_start_time
            remaining_sleep = sleep_duration - elapsed_time
            if remaining_sleep > 0:
                sleep_time = min(remaining_sleep, deadline - time.time())
                if sleep_time > 0:
                    time.sleep(sleep_time)
        return current_ui_elements_state

    def get_orientation(self) -> int:
        """Returns the current screen orientation.
        0: natural, 1: left, 2: right, 3: upside down.
        """
        return self.u2d.info.get("displayRotation", 0)

    def get_physical_frame_boundary(self) -> tuple[int, int, int, int]:
        """Returns the physical frame boundary.

        Args:
            env: The AndroidEnv interface.

        Returns:
            First two integers are the coordinates for top left corner, last two are for
            lower right corner. All coordinates are given in portrait orientation.
        """
        response = self.run_shell_command("dumpsys input | grep physicalFrame")
        if response.exit_code == 0:
            raw_output = response.output
            pattern = r"physicalFrame=\[(\d+), (\d+), (\d+), (\d+)\]"
            matches = re.findall(pattern, raw_output)
            orientation = self.get_orientation()
            for m in matches:
                if (
                    int(m[0]) == 0
                    and int(m[1]) == 0
                    and int(m[2]) == 0
                    and int(m[3]) == 0
                ):
                    continue
                if orientation == 0 or orientation == 2:
                    return (int(m[0]), int(m[1]), int(m[2]), int(m[3]))
                return (int(m[1]), int(m[0]), int(m[3]), int(m[2]))
        raise ValueError("Failed to get physical frame boundary.")


def is_element_available(ele: UIElement) -> bool:
    """
    Check if the UI element is available for interaction.
    """
    return (
        ele.is_clickable
        or ele.is_scrollable
        or ele.is_long_clickable
        or ele.is_editable
    )


def is_element_need_to_explore(ele: UIElement) -> bool:
    """
    Check if the element needs to be explored.
    """
    return ele.resource_id not in {
        "com.android.systemui:id/back",
        "com.android.systemui:id/recent_apps",
        "com.android.systemui:id/home",
        "com.android.systemui:id/home_button",
        "com.android.systemui:id/quick_settings",
        "com.android.systemui:id/ime_switcher",
        "com.android.chrome:id/sync_promo_signin_button",
        "com.android.systemui:id/menu_container",
        "com.android.settings:id/content_parent",
        "com.android.settings:id/recycler_view",
        "android:id/hard_keyboard_switch",
    }


import logging
import copy
import time

try:
    from . import json_action
except ImportError:
    import json_action


def send_android_intent(
    command: str,
    action: str,
    device_controller: Device,
    data_uri: str | None = None,
    mime_type: str | None = None,
    extras: dict[str, Any] | None = None,
    timeout_sec: int = 10,
):
    """Sends an intent to Android device using adb.

    This is a low-level command for sending an intent with additional parameters.
    When these additional parameters are not necessary, consider instead using
    `adb_utils.start_activity()` or `env.execute_adb_call()` with
    `AdbRequest.StartActivity` or `AdbRequest.SendBroadcast`.

    Args:
      command: Either "start" for start activity intents or "broadcast" for
        broadcast intents.
      action: The broadcast action (e.g. "android.intent.action.VIEW").
      env: The environment to which the broadcast is sent.
      data_uri: Optional intent data URI (e.g. "content://contacts/people/1").
      mime_type: Optional mime type (e.g. "image/png").
      extras: Dictionary containing keys and values to be sent as extras.
      timeout_sec: The maximum time in seconds to wait for the broadcast to
        complete.

    Returns:
      AdbResponse object.
    """
    if command not in ["start", "broadcast"]:
        raise ValueError('Intent command must be either "start" or "broadcast"')

    adb_command = ["am", command, "-a", action]

    if data_uri:
        adb_command.extend(["-d", f'"{data_uri}"'])

    if mime_type:
        adb_command.extend(["-t", f'"{mime_type}"'])

    if extras:
        for key, value in extras.items():
            if value is tuple:
                type_override, value = value
                if type_override == "str":
                    adb_command.extend(["--es", key, f'"{value}"'])
                elif type_override == "bool":
                    adb_command.extend(["--ez", key, f'"{value}"'])
                elif type_override == "int":
                    adb_command.extend(["--ei", key, f'"{value}"'])
                elif type_override == "long":  # long type only available via override.
                    adb_command.extend(["--el", key, f'"{value}"'])
                elif type_override == "float":
                    adb_command.extend(["--ef", key, f'"{value}"'])
                elif type_override == "string array":
                    array_str = ",".join(value)
                    adb_command.extend(["--esa", key, f'"{array_str}"'])
            elif isinstance(value, str):
                adb_command.extend(["--es", key, f'"{value}"'])
            elif isinstance(value, bool):
                adb_command.extend(["--ez", key, f'"{value}"'])
            elif isinstance(value, int):
                adb_command.extend(["--ei", key, f'"{value}"'])
            elif isinstance(value, float):
                adb_command.extend(["--ef", key, f'"{value}"'])
            elif isinstance(value, list):
                array_str = ",".join(value)
                adb_command.extend(["--esa", key, f'"{array_str}"'])
            else:
                raise ValueError(f"Unrecognized extra type for {key}")

    return device_controller.run_shell_command(adb_command, timeout=timeout_sec)


def display_message(message: str, device_controller, header: str = "") -> None:
    send_android_intent(
        command="broadcast",
        action="com.example.ACTION_UPDATE_OVERLAY",
        device_controller=device_controller,
        extras={"task_type_string": header, "goal_string": message},
    )


def execute_adb_action(
    action: json_action.JSONAction,
    device_controller: Device,
    screen_elements: list[Any] = None,  # list[UIElement]
    screen_size: tuple[int, int] = None,  # (width, height)
) -> None:
    """Execute an action based on a JSONAction object.

    Args:
        action: JSONAction object containing the action to be executed.
        screen_elements: List of UI elements on the screen.
        screen_size: The (width, height) of the screen.
        env: The environment to execute the action in.
    """
    if action.action_type == json_action.ANSWER:
        if action.text:
            display_message(
                action.text,
                header="Agent answered:",
                device_controller=device_controller,
            )
        return
    if action.action_type in ["click", "double_tap", "long_press"]:
        idx = action.index
        x = action.x
        y = action.y
        if idx is not None and screen_elements is not None:
            if idx < 0 or idx >= len(screen_elements):
                raise ValueError(
                    f"Invalid element index: {idx}, must be between 0 and"
                    f" {len(screen_elements)-1}."
                )
            element = screen_elements[idx]
            if element.bbox_pixels is None:
                raise ValueError("Bbox is not present on element.")
            x, y = element.bbox_pixels.center
            x, y = int(x), int(y)
            if action.action_type == "click":
                device_controller.click(x, y)
            elif action.action_type == "double_tap":
                device_controller.double_click(x, y)
            else:
                device_controller.long_click(x, y)
        elif x is not None and y is not None:
            if action.action_type == "click":
                device_controller.click(x, y)
            elif action.action_type == "double_tap":
                device_controller.double_click(x, y)
            else:
                device_controller.long_click(x, y)
        else:
            raise ValueError(f"Invalid click action: {action}")
        return x, y

    elif action.action_type == "input_text":
        text = action.text
        if text:
            click_action = copy.deepcopy(action)
            click_action.action_type = "click"
            execute_adb_action(
                action=click_action,
                device_controller=device_controller,
                screen_elements=screen_elements,
                screen_size=screen_size,
            )
            time.sleep(1.0)
            device_controller.input_text(text, smart_enter=True, clear_first=True)
        else:
            logging.warning(
                "Input_text action indicated, but no text provided. No "
                "action will be executed."
            )

    elif action.action_type == "keyboard_enter":
        device_controller.enter()

    elif action.action_type == "navigate_home":
        device_controller.home()

    elif action.action_type == "navigate_back":
        device_controller.back()

    elif action.action_type == "scroll":
        screen_width, screen_height = screen_size
        if action.index:
            x_min, y_min, x_max, y_max = (
                max(screen_elements[action.index].bbox_pixels.x_min, 0),
                max(screen_elements[action.index].bbox_pixels.y_min, 0),
                min(screen_elements[action.index].bbox_pixels.x_max, screen_width),
                min(screen_elements[action.index].bbox_pixels.y_max, screen_height),
            )
        else:
            x_min, y_min, x_max, y_max = (0, 0, screen_width, screen_height)

        start_x, start_y = (x_min + x_max) // 2, (y_min + y_max) // 2
        direction = action.direction
        if direction == "down":
            end_x, end_y = (x_min + x_max) // 2, y_min
        elif direction == "up":
            end_x, end_y = (x_min + x_max) // 2, y_max
        elif direction == "right":
            end_x, end_y = x_min, (y_min + y_max) // 2
        elif direction == "left":
            end_x, end_y = x_max, (y_min + y_max) // 2
        else:
            print("Invalid direction")
            return
        device_controller.swipe(int(start_x), int(start_y), int(end_x), int(end_y))
        return int(start_x), int(start_y), int(end_x), int(end_y)

    elif action.action_type == "swipe":  # Inverse of scroll.
        screen_width, screen_height = screen_size
        mid_x, mid_y = 0.5 * screen_width, 0.5 * screen_height
        direction = action.direction
        if direction == "down":
            start_x, start_y = mid_x, 0
            end_x, end_y = mid_x, screen_height
        elif direction == "up":
            start_x, start_y = mid_x, screen_height
            end_x, end_y = mid_x, 0
        elif direction == "left":
            start_x, start_y = 0, mid_y
            end_x, end_y = screen_width, mid_y
        elif direction == "right":
            start_x, start_y = screen_width, mid_y
            end_x, end_y = 0, mid_y
        else:
            print("Invalid direction")
            return
        device_controller.swipe(
            int(start_x), int(start_y), int(end_x), int(end_y), duration=0.5
        )
        return int(start_x), int(start_y), int(end_x), int(end_y)

    elif action.action_type == "open_app":
        app_name = action.app_name
        if app_name:
            launch_app(app_name, device_controller)
        else:
            raise ValueError("No app name provided")

    elif action.action_type == "wait":
        time.sleep(2.0)

    elif action.action_type == "launch_adb_activity":
        if action.activity_nickname == "app_drawer":
            device_controller.home()
            time.sleep(1.0)
            start_x, start_y = int(screen_size[0] / 2), int(screen_size[1] * 0.9)
            end_x = start_x
            end_y = int(0.3 * screen_size[1])
            device_controller.swipe(start_x, start_y, end_x, end_y)
        elif action.activity_nickname == "quick_settings":
            start_x, start_y = int(screen_size[0] / 2), 30
            end_x = start_x
            end_y = int(0.3 * screen_size[1])
            device_controller.swipe(start_x, start_y, end_x, end_y, duration=0.1)
    elif action.action_type == "change_orientation":
        change_orientation(action.orientation, device_controller)
    elif action.action_type == json_action.UNKNOWN:
        print("Unknown action type; no action will be executed. Try again...")
    else:
        print("Invalid action type")


def launch_app(app_name: str, device_controller: Device) -> Optional[str]:
    """Uses regex and ADB activity to try to launch an app.

    Args:
      app_name: The name of the app, as represented as a key in
        _PATTERN_TO_ACTIVITY.
      device_controller: The device controller to execute the command.

    Returns:
      The name of the app that is launched.
    """
    # Maps app names to the activity that should be launched to open the app.
    _PATTERN_TO_ACTIVITY = {
        "google chrome|chrome": (
            "com.android.chrome/com.google.android.apps.chrome.Main"
        ),
        "google chat": "com.google.android.apps.dynamite/com.google.android.apps.dynamite.startup.StartUpActivity",
        "settings|system settings": "com.android.settings/.Settings",
        "youtube|yt": "com.google.android.youtube/com.google.android.apps.youtube.app.WatchWhileActivity",
        "google play|play store|gps": (
            "com.android.vending/com.google.android.finsky.activities.MainActivity"
        ),
        "gmail|gemail|google mail|google email|google mail client": (
            "com.google.android.gm/.ConversationListActivityGmail"
        ),
        "google maps|gmaps|maps|google map": (
            "com.google.android.apps.maps/com.google.android.maps.MapsActivity"
        ),
        "google photos|gphotos|photos|google photo|google pics|google images": "com.google.android.apps.photos/com.google.android.apps.photos.home.HomeActivity",
        "google calendar|gcal": (
            "com.google.android.calendar/com.android.calendar.AllInOneActivity"
        ),
        "camera": "com.android.camera2/com.android.camera.CameraLauncher",
        "audio recorder": "com.dimowner.audiorecorder/com.dimowner.audiorecorder.app.welcome.WelcomeActivity",
        "google drive|gdrive|drive": (
            "com.google.android.apps.docs/.drive.startup.StartupActivity"
        ),
        "google keep|gkeep|keep": (
            "com.google.android.keep/.activities.BrowseActivity"
        ),
        "grubhub": (
            "com.grubhub.android/com.grubhub.dinerapp.android.splash.SplashActivity"
        ),
        "tripadvisor": "com.tripadvisor.tripadvisor/com.tripadvisor.android.ui.launcher.LauncherActivity",
        "starbucks": "com.starbucks.mobilecard/.main.activity.LandingPageActivity",
        "google docs|gdocs|docs": "com.google.android.apps.docs.editors.docs/com.google.android.apps.docs.editors.homescreen.HomescreenActivity",
        "google sheets|gsheets|sheets": "com.google.android.apps.docs.editors.sheets/com.google.android.apps.docs.editors.homescreen.HomescreenActivity",
        "google slides|gslides|slides": "com.google.android.apps.docs.editors.slides/com.google.android.apps.docs.editors.homescreen.HomescreenActivity",
        "clock": "com.google.android.deskclock/com.android.deskclock.DeskClock",
        "google search|google": "com.google.android.googlequicksearchbox/com.google.android.googlequicksearchbox.SearchActivity",
        "contacts": "com.google.android.contacts/com.android.contacts.activities.PeopleActivity",
        "facebook|fb": "com.facebook.katana/com.facebook.katana.LoginActivity",
        "whatsapp|wa": "com.whatsapp/com.whatsapp.Main",
        "instagram|ig": (
            "com.instagram.android/com.instagram.mainactivity.MainActivity"
        ),
        "twitter|tweet": "com.twitter.android/com.twitter.app.main.MainActivity",
        "snapchat|sc": "com.snapchat.android/com.snap.mushroom.MainActivity",
        "telegram|tg": "org.telegram.messenger/org.telegram.ui.LaunchActivity",
        "linkedin": (
            "com.linkedin.android/com.linkedin.android.authenticator.LaunchActivity"
        ),
        "spotify|spot": "com.spotify.music/com.spotify.music.MainActivity",
        "netflix": "com.netflix.mediaclient/com.netflix.mediaclient.ui.launch.UIWebViewActivity",
        "amazon shopping|amazon|amzn": (
            "com.amazon.mShop.android.shopping/com.amazon.mShop.home.HomeActivity"
        ),
        "tiktok|tt": "com.zhiliaoapp.musically/com.ss.android.ugc.aweme.splash.SplashActivity",
        "discord": "com.discord/com.discord.app.AppActivity$Main",
        "reddit": "com.reddit.frontpage/com.reddit.frontpage.MainActivity",
        "pinterest": "com.pinterest/com.pinterest.activity.PinterestActivity",
        "android world": "com.example.androidworld/.MainActivity",
        "files": "com.google.android.documentsui/com.android.documentsui.files.FilesActivity",
        "markor": "net.gsantner.markor/net.gsantner.markor.activity.MainActivity",
        "clipper": "ca.zgrs.clipper/ca.zgrs.clipper.Main",
        "messages": "com.google.android.apps.messaging/com.google.android.apps.messaging.ui.ConversationListActivity",
        "simple sms messenger|simple sms|sms messenger": "com.simplemobiletools.smsmessenger/com.simplemobiletools.smsmessenger.activities.MainActivity",
        "dialer|phone": "com.google.android.dialer/com.google.android.dialer.extensions.GoogleDialtactsActivity",
        "simple calendar pro|simple calendar": "com.simplemobiletools.calendar.pro/com.simplemobiletools.calendar.pro.activities.MainActivity",
        "simple gallery pro|simple gallery": "com.simplemobiletools.gallery.pro/com.simplemobiletools.gallery.pro.activities.MainActivity",
        "miniwob": "com.google.androidenv.miniwob/com.google.androidenv.miniwob.app.MainActivity",
        "simple draw pro": "com.simplemobiletools.draw.pro/com.simplemobiletools.draw.pro.activities.MainActivity",
        "pro expense|pro expense app": (
            "com.arduia.expense/com.arduia.expense.ui.MainActivity"
        ),
        "broccoli|broccoli app|broccoli recipe app|recipe app": (
            "com.flauschcode.broccoli/com.flauschcode.broccoli.MainActivity"
        ),
        "caa|caa test|context aware access": "com.google.ccc.hosted.contextawareaccess.thirdpartyapp/.ChooserActivity",
        "osmand": "net.osmand/net.osmand.plus.activities.MapActivity",
        "tasks|tasks app|tasks.org:": (
            "org.tasks/com.todoroo.astrid.activity.MainActivity"
        ),
        "open tracks sports tracker|activity tracker|open tracks|opentracks": (
            "de.dennisguse.opentracks/de.dennisguse.opentracks.TrackListActivity"
        ),
        "joplin|joplin app": "net.cozic.joplin/.MainActivity",
        "vlc|vlc app|vlc player": "org.videolan.vlc/.gui.MainActivity",
        "retro music|retro|retro player": (
            "code.name.monkey.retromusic/.activities.MainActivity"
        ),
    }

    def get_adb_activity(app_name: str) -> Optional[str]:
        """Get a mapping of regex patterns to ADB activities top Android apps."""
        for pattern, activity in _PATTERN_TO_ACTIVITY.items():
            if re.match(pattern.lower(), app_name.lower()):
                return activity

    # Special app names that will trigger opening the default app.
    _DEFAULT_URIS: dict[str, str] = {
        "calendar": "content://com.android.calendar",
        "browser": "http://",
        "contacts": "content://contacts/people/",
        "email": "mailto:",
        "gallery": "content://media/external/images/media/",
    }
    if app_name in _DEFAULT_URIS:
        data_uri = _DEFAULT_URIS[app_name]
        adb_command = [
            #'shell',
            "am",
            "start",
            "-a",
            "android.intent.action.VIEW",
            "-d",
            data_uri,
        ]
        device_controller.run_shell_command(adb_command)
        time.sleep(1.0)  # Wait for the app to launch.
        return app_name

    activity = get_adb_activity(app_name)
    if activity is None:
        logging.error("Failed to launch app: %r", app_name)
        return None
    args = [
        "am",
        "start",
        "-a",
        "android.intent.action.MAIN",
        "-c",
        "android.intent.category.LAUNCHER",
        "-n",
        activity,
    ]
    device_controller.run_shell_command(args)
    time.sleep(1.0)  # Wait for the app to launch.
    return app_name


def change_orientation(orientation: str, device_controller: Device) -> None:
    """Changes the screen orientation.

    Args:
      orientation: str, The new orientation. Can be portrait, landscape,
        reverse_portrait, or reverse_landscape.
      device_controller: The device controller to execute the command.

    Raises:
      ValueError if invalid orientation is provided.
    """
    _ORIENTATIONS = {
        "portrait": "0",
        "landscape": "1",
        "portrait_reversed": "2",
        "landscape_reversed": "3",
    }
    if orientation not in _ORIENTATIONS:
        raise ValueError(
            f"Unknown orientation provided: {orientation} not in"
            f" {_ORIENTATIONS.keys()}"
        )
    command = [
        "settings",
        "put",
        "system",
    ]
    device_controller.run_shell_command(command + ["accelerometer_rotation", "0"])
    device_controller.run_shell_command(
        command + ["user_rotation", _ORIENTATIONS[orientation]]
    )
