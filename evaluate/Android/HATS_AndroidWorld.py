from android_world.agents import base_agent
from android_world.env import interface
import re
from dataclasses import asdict

try:
    from . import json_action
except ImportError:
    import json_action

from .device import (
    Device,
    np_array_to_jpeg_base64,
    openai_request,
    extract_json,
    resize_ndarray_image,
    validate_ui_element,
    UIElement,
    execute_adb_action,
    resize_ndarray_image_qwen,
)
import copy
import numpy as np
from typing import Optional, Union, Dict


def scale_ui_element_bbox(ele: UIElement):
    bbox = ele.bbox
    if isinstance(bbox.x_min, int):
        return  # 如果是int类型，说明已经是缩放后的值了
    ele.bbox.x_min = max(min(int(bbox.x_min * 1000), 1000), 0)
    ele.bbox.x_max = max(min(int(bbox.x_max * 1000), 1000), 0)
    ele.bbox.y_min = max(min(int(bbox.y_min * 1000), 1000), 0)
    ele.bbox.y_max = max(min(int(bbox.y_max * 1000), 1000), 0)


def _generate_ui_element_description(ui_element: UIElement, index: int = None) -> str:
    """Generate a description for a given UI element with important information.

    Args:
      ui_element: UI elements for the current screen.
      index: The numeric index for the UI element.

    Returns:
      The description for the UI element.
    """
    element_description = (
        f"UI element {index}: {{" if index is not None else "UI element: {"
    )
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

    if hasattr(
        ui_element, "hints"
    ):
        if ui_element.hints:
            element_description += f'"hints": {ui_element.hints}, '

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
    if ui_element.bbox:
        scale_ui_element_bbox(ui_element)
        x, y = ui_element.bbox.center
        x, y = int(x), int(y)
        element_description += f'"x": {x}, "y": {y}, '

    return element_description[:-2] + "}"  # 这里的[:-2]是为了去掉', '


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
    a11y_tree = ""
    for index, ui_element in enumerate(ui_elements):
        if validate_ui_element(ui_element, screen_width_height_px) and ui_element.bbox:
            a11y_tree += _generate_ui_element_description(ui_element, index) + "\n"
    return a11y_tree.strip()



REASONING = """You are a GUI task expert, I will provide you with a high-level instruction, an action history, a screenshot with its corresponding accessibility tree.

High-level instruction:
    ```
    {high_level_instruction}
    ```

Action history:
    ```
    {action_history}
    ```

Accessibility tree:
    ```
    {accessibility_tree}
    ```

Please generate the low-level thought and action for the next step.
"""


def parse_reason_action_output(
    raw_reason_action_output: str,
) -> tuple[Optional[str], Optional[str]]:
    r"""Parses llm action reason output.

    Args:
      raw_reason_action_output: Raw string output that supposes to have the format
        'Low-level thought: xxx\nAction:xxx'.

    Returns:
      If parsing successfully, returns reason and action.
    """
    reason_result = re.search(
        r"thought:(.*)Action:", raw_reason_action_output, flags=re.DOTALL
    )
    reason = reason_result.group(1).strip() if reason_result else None
    action_result = re.search(r"Action:(.*)", raw_reason_action_output, flags=re.DOTALL)
    action = action_result.group(1).strip() if action_result else None
    return reason, action


def generate_action_text(converted_action, ele: UIElement = None) -> str:
    if ele is None:
        return converted_action.json_str()
    scale_ui_element_bbox(ele)
    x, y = ele.bbox.center
    converted_action.x = int(x)
    converted_action.y = int(y)
    converted_action.index = None
    return converted_action.json_str()


import copy


def generate_action_history(
    history_step_cnt: int,
    execution_trajectory_data: list,
    screen_width: int,
    screen_height: int,
) -> str:
    if history_step_cnt == 0:
        return "None"
    action_history = ""
    step_idx = 0
    for data in execution_trajectory_data[:history_step_cnt]:
        converted_action = copy.deepcopy(data["converted_action"])
        if isinstance(converted_action, str):
            continue
        if converted_action.x is not None and converted_action.y is not None:
            converted_action.x = min(
                1000, max(0, int(converted_action.x * 1000 / screen_width))
            )
            converted_action.y = min(
                1000, max(0, int(converted_action.y * 1000 / screen_height))
            )

        step_idx += 1
        ele = data["target_element"]  # asdict(UIElement) or None
        if ele is not None:
            ele = UIElement(**ele)
        action_history += f"Step {step_idx}:\n"
        action_history += f"thought: {data['reasoning']}\n"
        action_history += f"Action: {generate_action_text(converted_action,ele)}\n"
    return action_history.strip()


class HATSAgent(base_agent.EnvironmentInteractingAgent):
    """
    HATS agent AndroidWorld
    """

    def __init__(
        self,
        env: interface.AsyncEnv,
        name: str = "HATSAgent",
        model: str = "internvl2_4b",
        openai_base_url: str = "http://localhost:8000/v1",
        documents: dict[str, str] = {},
    ) -> None:
        super().__init__(env, name)
        self.device = Device(
            f"emulator-{env.controller.env._coordinator._simulator._config.emulator_launcher.emulator_console_port}"
        )
        self.screen_width, self.screen_height = self.device.get_screen_size()
        self.history = []
        self.last_goal = None
        self.model = model
        self.openai_base_url = openai_base_url
        self.documents = documents

    def reset(self, go_home: bool = False) -> None:
        super().reset(go_home)
        self.env.hide_automation_ui()
        self.history = []
        self.last_goal = None

    def step(
        self, goal: str, verbose: bool = True
    ) -> base_agent.AgentInteractionResult:
        if self.last_goal is None or self.last_goal != goal:
            self.history = []
            self.last_goal = goal
        step_data = {
            "raw_screenshot": None,
            "reasoning": "None",
            "action_output": "None",
            "converted_action": "error_retry",
            "actual_action_coordinates": None,
            "benchmark_screenshot": None,
            "ui_elements": None,
            "top_app_package_name": None,
            "target_element": None,
        }
        print("\n----------step " + str(len(self.history) + 1))

        before_ui_elements = self.device.wait_to_stabilize()
        logical_screen_size = self.device.get_screen_size()

        for ele in before_ui_elements:
            if ele.uid in self.documents:
                ele.hints = self.documents[ele.uid]

        step_data["ui_elements"] = [
            asdict(ui_element) for ui_element in before_ui_elements
        ]
        accessibility_tree = _generate_ui_elements_description_list(
            before_ui_elements, logical_screen_size
        )
        before_screenshot = np.array(self.device.get_screenshot())
        step_data["raw_screenshot"] = before_screenshot.copy()
        step_data["benchmark_screenshot"] = copy.deepcopy(before_screenshot)
        top_app_package_name = self.device.get_top_package_name()
        step_data["top_app_package_name"] = top_app_package_name
        action_history = ""
        action_history = generate_action_history(
            len(self.history), self.history, self.screen_width, self.screen_height
        )
        text_prompt = REASONING.format(
            high_level_instruction=goal,
            action_history=action_history,
            accessibility_tree=accessibility_tree,
        )

        screenshot = before_screenshot
        if "qwen" in self.model:
            screenshot = resize_ndarray_image(screenshot, target_max_size=1024)
        elif "internvl" in self.model:
            screenshot = resize_ndarray_image(screenshot, target_max_size=1024)
        else:
            raise ValueError(f"Unsupported model {self.model} for resizing screenshot.")
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{np_array_to_jpeg_base64(screenshot)}",
                        },
                    },
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]

        rsp_text = openai_request(
            messages=messages,
            model=self.model,
            openai_api_key="",
            openai_base_url=self.openai_base_url,
            temperature=0.0,
            max_tokens=1024,
            timeout=600,
        )
        reason, action = parse_reason_action_output(rsp_text)
        action = action.replace("double_click", "double_tap")
        action = action.replace('''," "''', ''',"''')
        print("Reasoning: ", reason)
        print("Action: ", action)
        step_data["reasoning"] = reason
        step_data["action_output"] = action
        if (not reason) or (not action):
            print("Action prompt output is not in the correct format.")
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(
                done=False,
                data=step_data,
            )

        try:
            converted_action = json_action.JSONAction(
                **extract_json(action),
            )
            x = converted_action.x
            y = converted_action.y
            if x is not None and y is not None:
                converted_action.x = min(
                    max(int(self.screen_width * x / 1000), 0), self.screen_width
                )
                converted_action.y = min(
                    max(int(self.screen_height * y / 1000), 0), self.screen_height
                )

            step_data["converted_action"] = converted_action
            if converted_action.index is not None:
                if converted_action.index < 0 or converted_action.index >= len(
                    before_ui_elements
                ):
                    raise ValueError(
                        f"Invalid index {converted_action.index} for UI elements."
                    )
                step_data["target_element"] = asdict(
                    before_ui_elements[converted_action.index]
                )
        except Exception as e:  # pylint: disable=broad-exception-caught
            print("Failed to convert the output to a valid action.")
            print(str(e))
            step_data["converted_action"] = "error_retry"
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(
                done=False,
                data=step_data,
            )

        if converted_action.action_type == "status":
            if converted_action.goal_status == "infeasible":
                print("Agent stopped since it thinks mission impossible.")
            else:
                print("Agent thinks the request has been completed.")
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(
                done=True,
                data=step_data,
            )

        if converted_action.action_type == "answer":
            print("Agent answered with: " + converted_action.text)

        try:
            actual_action_coordinates = execute_adb_action(
                converted_action,
                self.device,
                before_ui_elements,
                logical_screen_size,
            )
            step_data["actual_action_coordinates"] = actual_action_coordinates
        except Exception as e:  # pylint: disable=broad-exception-caught
            print("Failed to execute action.")
            print(str(e))
            step_data["converted_action"] = "error_retry"
            self.history.append(step_data)
            return base_agent.AgentInteractionResult(
                done=False,
                data=step_data,
            )
        self.history.append(step_data)
        return base_agent.AgentInteractionResult(
            done=False,
            data=step_data,
        )
