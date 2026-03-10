from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Union
import json
from device import (
    Device,
    is_element_available,
    UIElement,
    is_element_need_to_explore,
    add_screenshot_label,
    add_ui_element_mark,
    _generate_ui_element_description,
)
import math
from utils import generate_input_text_for_editable_element
import numpy as np
import random
from prompt_templates import (
    COUNT_EXPLORATION_STEPS_MATCHED_PROMPT,
    REFINE_HIGH_LEVEL_INSTRUCTION_PROMPT,
)
from utils import openai_request, extract_json

random_seed = 42  # 随机种子
random_obj = random.Random(random_seed)

UCT_CONST = 1.414  # UCT常数，通常取2^0.5，表示探索和利用之间的权衡。
IS_VISITED_NODE = {}  # k:node uid, v:True


@dataclass
class MCTS_Node:
    """
    A class representing a node in the Monte Carlo Tree Search (MCTS) algorithm.
    """

    parent: Optional["str"] = None
    children: List["str"] = None
    visits: int = 0
    score: float = 0.0
    uid: str = None
    ele: UIElement = None
    actions: Dict[str, Any] = None

    def __eq__(self, other: "MCTS_Node") -> bool:
        """
        Check if two MCTS nodes are equal based on their unique identifiers.
        """
        assert isinstance(other, MCTS_Node), f"other is {type(other)}"
        return self.uid == other.uid

    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.ele:
            assert self.ele.uid is not None, f"ele.uid is {self.ele.uid}"
            self.uid = self.ele.uid
        if self.actions is None:
            self.actions = {
                "click": {
                    "is_available": self.ele.is_clickable if self.ele else False,
                    "actions": {
                        "click": {"visits": 0, "score": 0.0},
                    },
                },
                "long_click": {
                    "is_available": self.ele.is_long_clickable if self.ele else False,
                    "actions": {
                        "long_click": {"visits": 0, "score": 0.0},
                    },
                },
                "scroll": {
                    "is_available": self.ele.is_scrollable if self.ele else False,
                    "actions": {
                        "scroll_up": {"visits": 0, "score": 0.0},
                        "scroll_down": {"visits": 0, "score": 0.0},
                        "scroll_left": {"visits": 0, "score": 0.0},
                        "scroll_right": {"visits": 0, "score": 0.0},
                    },
                },
                "input": {
                    "is_available": self.ele.is_editable if self.ele else False,
                    "input_content": None,
                    "actions": {
                        "input": {"visits": 0, "score": 0.0},
                    },
                },
            }

    def _try_to_generate_input_content(self):
        """
        Try to generate input content for the node's actions.
        """
        if self.ele and self.ele.is_editable:
            if self.ele.text:
                self.actions["input"]["input_content"] = self.ele.text
            elif self.ele.hint_text:
                self.actions["input"]["input_content"] = self.ele.hint_text

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the MCTS node to a dictionary representation.
        """
        return {
            "uid": self.uid,
            "parent": self.parent,
            "visits": self.visits,
            "score": self.score,
            "actions": self.actions,
            "children": self.children,
        }

    def __str__(self) -> str:
        """
        Convert the MCTS node to a string representation.
        """
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


def flatten_node_actions(node: MCTS_Node) -> List[Tuple[str, str, str]]:
    """
    Flatten the actions of a node into a list of tuples.

    Each tuple contains the node's unique identifier, action type, and action name.
    """
    ret = []
    for action_type, v in node.actions.items():
        if v["is_available"]:
            for action_name, d in v["actions"].items():
                nid = f"{node.uid}_{action_type}_{action_name}"
                if nid in IS_VISITED_NODE:
                    continue
                ret.append((node.uid, action_type, action_name))
    return ret


def uct(action_tuple: Tuple[str, str, str], tree_nodes: Dict[str, MCTS_Node]) -> float:
    """
    Calculate the Upper Confidence Bound for Trees (UCT) value for a given action tuple.
    """
    uid, action_type, action_name = action_tuple
    node = tree_nodes[uid]
    nid = f"{node.uid}_{action_type}_{action_name}"
    if nid in IS_VISITED_NODE:
        return -1
    parent = tree_nodes[node.parent] if node.parent else None
    parent_visits = parent.visits if parent else 1
    score = node.actions[action_type]["actions"][action_name]["score"]
    visits = node.actions[action_type]["actions"][action_name]["visits"]
    return score + UCT_CONST * math.sqrt(math.log(parent_visits) / (1 + visits))


def get_candidate_nodes_for_selection(
    eles: List[UIElement], parent_uid: str, tree_nodes: Dict[str, MCTS_Node]
) -> List[MCTS_Node]:
    """
    Get candidate nodes for expansion based on the available UI elements.
    """
    candidate_nodes = []
    for ele in eles:
        if is_element_available(ele) and is_element_need_to_explore(ele):
            node = MCTS_Node(ele=ele, parent=parent_uid)
            if node.uid in tree_nodes:
                candidate_nodes.append(tree_nodes[node.uid])
    return candidate_nodes


def selection(
    nodes: List[MCTS_Node], tree_nodes: Dict[str, MCTS_Node]
) -> List[Tuple[str, str, str]]:
    """
    Select the best action based on the UCT value for a list of nodes.
    returns the action tuple (uid, action_type, action_name).
    """
    flatten_actions = []
    for node in nodes:
        if is_element_need_to_explore(node.ele):
            flatten_actions.extend(flatten_node_actions(node))
    if not flatten_actions:
        return None
    selected_action = max(flatten_actions, key=lambda action: uct(action, tree_nodes))
    nid = f"{selected_action[0]}_{selected_action[1]}_{selected_action[2]}"
    if nid in IS_VISITED_NODE:
        return None
    IS_VISITED_NODE[nid] = True
    return selected_action


def get_random_action(
    eles: List[UIElement],
    parent_uid: str,
    tree_nodes: Dict[str, MCTS_Node],
    insert_node: bool = False,
    max_branching_factor: int = 5,
) -> Union[Tuple[None, None], Tuple[Tuple[str, str, str], MCTS_Node]]:
    candidate_nodes = []
    for ele in eles:
        if is_element_available(ele) and is_element_need_to_explore(ele):
            node = MCTS_Node(ele=ele, parent=parent_uid)
            if node.uid not in tree_nodes:
                candidate_nodes.append(node)
            else:
                node = tree_nodes[node.uid]
                added = False
                for d in node.actions.values():
                    actions = d["actions"]
                    if added:
                        break
                    for action_name, action_info in actions.items():
                        if action_info["visits"] == 0:
                            candidate_nodes.append(node)
                            added = True
                            break
    if not candidate_nodes:
        return None, None
    selected_nodes = random_obj.sample(
        candidate_nodes, min(max_branching_factor, len(candidate_nodes))
    )
    selected_one_node = random_obj.choice(selected_nodes)
    if insert_node:
        if parent_uid in tree_nodes:
            for selected_node in selected_nodes:
                if selected_node.uid not in tree_nodes[parent_uid].children:
                    tree_nodes[parent_uid].children.append(selected_node.uid)
                    tree_nodes[selected_node.uid] = selected_node
    flattened_actions = flatten_node_actions(selected_one_node)
    if not flattened_actions or len(flattened_actions) == 0:
        return None, None
    selected_action = random_obj.choice(flattened_actions)
    return (
        selected_action,
        selected_one_node,
    )


def expansion(
    eles: List[UIElement],
    parent_uid: str,
    tree_nodes: Dict[str, MCTS_Node],
    max_branching_factor: int = 5,
) -> List[Tuple[str, str, str]]:
    selected_action, _ = get_random_action(
        eles=eles,
        parent_uid=parent_uid,
        tree_nodes=tree_nodes,
        insert_node=True,
        max_branching_factor=max_branching_factor,
    )
    if not selected_action:
        return None
    return selected_action


def execute_mcts_action(
    action: Tuple[str, str, str],
    device_controller: Device,
    node: Optional[MCTS_Node] = None,
    tree_nodes: Optional[Dict[str, MCTS_Node]] = None,
    usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0},
    update_visits: bool = True,
) -> None:
    uid, action_type, action_name = action
    assert (
        node is not None or tree_nodes is not None
    ), "node and tree_nodes are both None"
    if tree_nodes is not None:
        if update_visits:
            tree_nodes[uid].visits += 1
            tree_nodes[uid].actions[action_type]["actions"][action_name]["visits"] += 1
        ele = tree_nodes[uid].ele
        node = tree_nodes[uid]
    ele = node.ele
    screen_width, screen_height = device_controller.get_screen_size()
    x_min, y_min, x_max, y_max = (
        max(ele.bbox_pixels.x_min, 0),
        max(ele.bbox_pixels.y_min, 0),
        min(ele.bbox_pixels.x_max, screen_width),
        min(ele.bbox_pixels.y_max, screen_height),
    )
    start_x, start_y = int((x_min + x_max) // 2), int((y_min + y_max) // 2)
    if action_name == "click":
        device_controller.click(start_x, start_y)
    elif action_name == "double_click":
        device_controller.double_click(start_x, start_y)
    elif action_name == "long_click":
        device_controller.long_click(start_x, start_y)
    elif action_name == "scroll_up":
        end_x, end_y = (x_min + x_max) // 2, y_min
        device_controller.swipe(int(start_x), int(start_y), int(end_x), int(end_y))
    elif action_name == "scroll_down":
        end_x, end_y = (x_min + x_max) // 2, y_max
        device_controller.swipe(int(start_x), int(start_y), int(end_x), int(end_y))
    elif action_name == "scroll_left":
        end_x, end_y = x_min, (y_min + y_max) // 2
        device_controller.swipe(int(start_x), int(start_y), int(end_x), int(end_y))
    elif action_name == "scroll_right":
        end_x, end_y = x_max, (y_min + y_max) // 2
        device_controller.swipe(int(start_x), int(start_y), int(end_x), int(end_y))
    elif action_name == "input":
        input_content = node.actions["input"]["input_content"]
        if input_content is None:
            ndarray_screenshot = np.array(device_controller.get_screenshot())
            input_content = generate_input_text_for_editable_element(
                marked_ndarray_screenshot=ndarray_screenshot,
                ui_element_mark="1",
                usage=usage,
            )
            node.actions["input"]["input_content"] = input_content
            if tree_nodes is not None:
                tree_nodes[uid].actions["input"]["input_content"] = input_content
        device_controller.click(start_x, start_y)
        device_controller.wait_to_stabilize()
        device_controller.input_text(input_content)
    else:
        raise ValueError(f"Unknown action name: {action_name}")


def simulation(
    device_controller: Device,
    tree_nodes: Dict[str, MCTS_Node],
    max_steps: int = 6,
    usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0},
):
    trajectory_data = {
        "screenshots": [],
        "screenshot_with_steps": [],
        "actions": [],
        "package_names": [],
        "low_level_instructions": [],
        "analyses": [],
        "ui_elements": [],
        "score": 0.0,
        "high_level_instruction": None,
    }
    screen_width, screen_height = device_controller.get_screen_size()
    physical_frame_boundary = device_controller.get_physical_frame_boundary()
    orientation = device_controller.get_orientation()
    device_controller.wait_to_stabilize()
    step_idx = 1
    for _ in range(max_steps):
        screenshot = device_controller.get_screenshot()
        ui_elements = device_controller._get_ui_elements()
        top_package_name = device_controller.get_top_package_name()
        selected_action, node = get_random_action(
            eles=ui_elements,
            parent_uid=None,
            tree_nodes=tree_nodes,
            insert_node=False,
        )
        if selected_action is None:
            break
        uid, action_type, action_name = selected_action
        ele = node.ele
        ndarray_screenshot = np.array(screenshot)
        add_screenshot_label(
            ndarray_screenshot,
            f"Step {step_idx}",
        )
        add_ui_element_mark(
            ndarray_screenshot,
            ele,
            step_idx,
            (screen_width, screen_height),
            physical_frame_boundary,
            orientation,
        )
        execute_mcts_action(
            selected_action,
            device_controller,
            node=node,
            usage=usage,
            update_visits=False,
        )
        device_controller.wait_to_stabilize()
        if screenshot == device_controller.get_screenshot():
            continue
        step_idx += 1
        trajectory_data["screenshots"].append(screenshot)
        trajectory_data["ui_elements"].append(ui_elements)
        trajectory_data["package_names"].append(top_package_name)
        trajectory_data["actions"].append(
            {"action_type": action_type, "action_name": action_name, "node": node}
        )
        trajectory_data["screenshot_with_steps"].append(ndarray_screenshot)
    ui_elements = device_controller._get_ui_elements()
    trajectory_data["ui_elements"].append(ui_elements)
    trajectory_data["package_names"].append(device_controller.get_top_package_name())
    screenshot = device_controller.get_screenshot()
    trajectory_data["screenshots"].append(screenshot)
    ndarray_screenshot = np.array(screenshot)
    add_screenshot_label(
        ndarray_screenshot,
        "Final",
    )
    trajectory_data["screenshot_with_steps"].append(ndarray_screenshot)
    IS_VISITED_NODE = {}
    return trajectory_data


def verifier(
    exploration_trajectory_data: Dict[str, Any],
    execution_trajectory_data: List[Dict[str, Any]],
    usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0},
) -> Tuple[float, List[str]]:
    exploration_trajectory_information = ""
    step_idx = 0
    for i in exploration_trajectory_data["selected_step_idx"]:
        step_idx += 1  # 从1开始
        action = exploration_trajectory_data["actions"][i]["action_name"]
        action_type = exploration_trajectory_data["actions"][i]["action_type"]
        node = exploration_trajectory_data["actions"][i]["node"]
        package_name = exploration_trajectory_data["package_names"][i]
        exploration_trajectory_information += f"Step {step_idx}:\n"
        exploration_trajectory_information += f"Low Level Instruction: {exploration_trajectory_data['low_level_instructions'][i].strip()}\n"
        if action_type == "input":
            input_content = node.actions["input"]["input_content"]
            exploration_trajectory_information += f"Action: {action} `{input_content}` into {_generate_ui_element_description(node.ele,i + 1)}\n"
        else:
            exploration_trajectory_information += f"Action: {action} on {_generate_ui_element_description(node.ele,i + 1)}\n"
        exploration_trajectory_information += f"Package Name: {package_name}\n\n"
    exploration_trajectory_information = (
        exploration_trajectory_information.strip()
    )  # 去掉最后的换行符

    gui_agent_trajectory_information = ""
    for i in range(len(execution_trajectory_data)):
        step_idx = i + 1  # 从1开始
        action = execution_trajectory_data[i]["converted_action"]
        target_element = (
            UIElement(**execution_trajectory_data[i]["target_element"])
            if execution_trajectory_data[i]["target_element"]
            else None
        )
        gui_agent_trajectory_information += f"Step {step_idx}:\n"
        gui_agent_trajectory_information += (
            f"Reasoning: {execution_trajectory_data[i]["reasoning"]}\n"
        )
        if target_element:
            gui_agent_trajectory_information += f"Action: {action} on {_generate_ui_element_description(target_element,action.index if hasattr(action, 'index') else step_idx)}\n"
        else:
            gui_agent_trajectory_information += f"Action: {action}\n"
        gui_agent_trajectory_information += (
            f"Package Name: {execution_trajectory_data[i]["top_app_package_name"]}\n"
        )
        gui_agent_trajectory_information += (
            f"Summary: {execution_trajectory_data[i]['summary']}\n\n"
        )
    gui_agent_trajectory_information = (
        gui_agent_trajectory_information.strip()
    )  # 去掉最后的换行符

    p = COUNT_EXPLORATION_STEPS_MATCHED_PROMPT.format(
        high_level_instruction=exploration_trajectory_data[
            "high_level_instruction"
        ].strip(),
        exploration_trajectory_information=exploration_trajectory_information,
        gui_agent_trajectory_information=gui_agent_trajectory_information,
    )
    messages = [{"role": "user", "content": p}]
    response_text = openai_request(messages=messages, max_tokens=1000, usage=usage)
    d = extract_json(response_text)
    assert d is not None, f"extract_json failed: {response_text}"
    assert "match_num" in d, f"match_num not in response_text: {response_text}"
    assert (
        "matched_exploration_id" in d
    ), f"matched_exploration_id not in response_text: {response_text}"
    assert (
        "matched_gui_agent_id" in d
    ), f"matched_gui_agent_id not in response_text: {response_text}"
    num = min(d["match_num"], len(exploration_trajectory_data["selected_step_idx"]))
    recall = max(0, num) / len(exploration_trajectory_data["selected_step_idx"])
    matched_low_level_instructions = [
        exploration_trajectory_data["low_level_instructions"][
            exploration_trajectory_data["selected_step_idx"][i - 1]
        ].strip()
        for i in d["matched_exploration_id"]
    ]
    matched_gui_agent_steps = [
        execution_trajectory_data[i - 1]["reasoning"] for i in d["matched_gui_agent_id"]
    ]
    return recall, matched_low_level_instructions, matched_gui_agent_steps


def refine_high_level_instruction(
    high_level_instruction: str,
    exploration_trajectory_data: Dict[str, Any],
    execution_trajectory_data: List[Dict[str, Any]],
    matched_low_level_instructions: List[str],
    matched_gui_agent_steps: List[str],
    usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0},
) -> str:
    exploration_trajectory_information = ""
    step_idx = 0
    for i in exploration_trajectory_data["selected_step_idx"]:
        step_idx += 1  # 从1开始
        action = exploration_trajectory_data["actions"][i]["action_name"]
        action_type = exploration_trajectory_data["actions"][i]["action_type"]
        node = exploration_trajectory_data["actions"][i]["node"]
        package_name = exploration_trajectory_data["package_names"][i]
        exploration_trajectory_information += f"Step {step_idx}:\n"
        exploration_trajectory_information += f"Low Level Instruction: {exploration_trajectory_data['low_level_instructions'][i].strip()}\n"
        if action_type == "input":
            input_content = node.actions["input"]["input_content"]
            exploration_trajectory_information += f"Action: {action} `{input_content}` into {_generate_ui_element_description(node.ele,i + 1)}\n"
        else:
            exploration_trajectory_information += f"Action: {action} on {_generate_ui_element_description(node.ele,i + 1)}\n"
        exploration_trajectory_information += f"Package Name: {package_name}\n\n"
    exploration_trajectory_information = exploration_trajectory_information.strip()

    gui_agent_trajectory_information = ""
    for i in range(len(execution_trajectory_data)):
        step_idx = i + 1  # 从1开始
        action = execution_trajectory_data[i]["converted_action"]
        target_element = (
            UIElement(**execution_trajectory_data[i]["target_element"])
            if execution_trajectory_data[i]["target_element"]
            else None
        )
        gui_agent_trajectory_information += f"Step {step_idx}:\n"
        gui_agent_trajectory_information += (
            f"Reasoning: {execution_trajectory_data[i]["reasoning"]}\n"
        )
        if target_element:
            gui_agent_trajectory_information += f"Action: {action} on {_generate_ui_element_description(target_element,action.index if hasattr(action, 'index') else step_idx)}\n"
        else:
            gui_agent_trajectory_information += f"Action: {action}\n"
        gui_agent_trajectory_information += (
            f"Package Name: {execution_trajectory_data[i]["top_app_package_name"]}\n"
        )
        gui_agent_trajectory_information += (
            f"Summary: {execution_trajectory_data[i]['summary']}\n\n"
        )
    gui_agent_trajectory_information = gui_agent_trajectory_information.strip()

    matched_low_level_instructions_str = ""
    for i in range(len(matched_low_level_instructions)):
        matched_low_level_instructions_str += (
            f"{i+1}. {matched_low_level_instructions[i]}\n"
        )
    matched_low_level_instructions_str = matched_low_level_instructions_str.strip()
    matched_gui_agent_steps_str = ""
    for i in range(len(matched_gui_agent_steps)):
        matched_gui_agent_steps_str += f"{i+1}. {matched_gui_agent_steps[i]}\n"
    matched_gui_agent_steps_str = matched_gui_agent_steps_str.strip()

    p = REFINE_HIGH_LEVEL_INSTRUCTION_PROMPT.format(
        high_level_instruction=high_level_instruction.strip(),
        exploration_trajectory_information=exploration_trajectory_information,
        gui_agent_trajectory_information=gui_agent_trajectory_information,
        matched_low_level_instructions=matched_low_level_instructions_str,
        matched_gui_agent_steps=matched_gui_agent_steps_str,
    )
    messages = [{"role": "user", "content": p}]
    response_text = openai_request(
        messages=messages,
        max_tokens=1000,
        usage=usage,
    )
    d = extract_json(response_text)
    assert d is not None, f"extract_json failed: {response_text}"
    refined_high_level_instruction = None
    if "refined_high_level_instruction" in d:
        refined_high_level_instruction = d["refined_high_level_instruction"].strip()
    elif "high_level_instruction" in d:
        refined_high_level_instruction = d["high_level_instruction"].strip()
    else:
        raise ValueError(f"refined_high_level_instruction not in d: {response_text}")
    return refined_high_level_instruction
