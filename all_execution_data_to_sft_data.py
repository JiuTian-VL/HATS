from utils import (
    load_object_from_disk,
    get_md5_hash,
    resize_pil_image,
    resize_ndarray_image,
)
import os
from device import UIElement
import re
import json
from mcts import MCTS_Node
from PIL import Image
from tqdm import tqdm
import random
import copy
import math
import shutil

random_seed = 42  # 随机种子
random_obj = random.Random(random_seed)

UCT_CONST = 1.414


def uct(score: float, visits: int, parent_visits: int) -> float:
    """
    Calculate the Upper Confidence Bound for Trees (UCT) value for a given action tuple.
    """
    return score + UCT_CONST * math.sqrt(math.log(parent_visits) / (1 + visits))


def extract_uid(s: str) -> str | None:
    """从文件名中提取出该轨迹的初始节点的父节点的uid"""
    pattern = r"\d{3}_(.*?)_execute_"
    match = re.search(pattern, s)
    if match:
        return (
            match.group(1)
            .split("_click")[0]
            .split("_long_click")[0]
            .split("_scroll")[0]
            .split("_input")[0]
        )
    return None


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


def generate_action_history(
    history_step_cnt: int, execution_trajectory_data: list
) -> str:
    if history_step_cnt == 0:
        return "None"
    action_history = ""
    step_idx = 0
    for data in execution_trajectory_data[:history_step_cnt]:
        converted_action = data["converted_action"]
        if isinstance(converted_action, str):
            continue
        ele = data["target_element"]  # asdict(UIElement) or None
        if ele is not None:
            ele = UIElement(**ele)
        if converted_action.index is not None and ele is None:
            print(
                f"skipping step {step_idx+1} due to missing target element. {converted_action}"
            )
            continue
        step_idx += 1
        action_history += f"Step {step_idx}:\n"
        action_history += f"thought: {data['reasoning']}\n"
        action_history += f"Action: {generate_action_text(converted_action,ele)}\n"
    return action_history.strip()


def generate_action_text(converted_action, ele: UIElement = None) -> str:
    if converted_action.index is None:
        return converted_action.json_str()
    assert ele is not None
    scale_ui_element_bbox(ele)
    x, y = ele.bbox.center
    converted_action.x = int(x)
    converted_action.y = int(y)
    converted_action.index = None
    return converted_action.json_str()


def generate_a11y_tree(ui_elements: list[UIElement]) -> str:
    a11y_tree = ""
    for i, ele in enumerate(ui_elements):
        a11y_tree += _generate_ui_element_description(ele, i) + "\n"
    return a11y_tree.strip()


# INPUT: high_level_instruction, action_history, a11y_tree
PROMPT_FOR_PLANNING_TRAINING_HUMAN = """<image>
You are a GUI task expert, I will provide you with a high-level instruction, an action history, a screenshot with its corresponding accessibility tree.

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
    {a11y_tree}
    ```

Please generate the low-level thought and action for the next step.
"""

# INPUT: low_level_thought, action
PROMPT_FOR_PLANNING_TRAINING_GPT = """Low-level thought: {low_level_thought}
Action: {action}
"""

# INPUT: action_history, a11y_tree, low_level_thought
PROMPT_FOR_ACTION_TRAINING_HUMAN = """<image>
You are a GUI task expert, I will provide you with an action history, a screenshot with its corresponding accessibility tree, and a low-level thought.

Action history:
    ```
    {action_history}
    ```

Accessibility tree:
    ```
    {a11y_tree}
    ```

Low-level thought:
    ```
    {low_level_thought}
    ```

Please generate the action for the next step."""

# INPUT: action
PROMPT_FOR_ACTION_TRAINING_GPT = """Action: {action}
"""


def _get_high_level_instruction_from_explore(execute_pkl_path: str) -> str:
    pkl_path = execute_pkl_path.replace("execute", "explore")
    pkl_dir = os.path.dirname(pkl_path)
    pkl_fn = os.path.basename(pkl_path)
    pkl_fn = "001" + pkl_fn[3:]
    pkl_path = os.path.join(pkl_dir, pkl_fn)
    trajectory_data = load_object_from_disk(pkl_path)
    high_level_instruction = None
    if "refined_high_level_instructions" in trajectory_data:
        high_level_instruction = trajectory_data["refined_high_level_instructions"][-1]
    elif "high_level_instruction" in trajectory_data:
        high_level_instruction = trajectory_data["high_level_instruction"]
    return high_level_instruction


import uuid

if __name__ == "__main__":
    # NOTE: 这里可以根据需要修改成其他路径
    mcts_dir = "mcts_output"
    # NOTE: 这里可以根据需要修改成其他路径
    dataset_root_dir = "mcts_dataset_execute"  # 数据集的根路径
    # NOTE: 这里可以根据需要修改
    max_sample_num = 1000

    shutil.rmtree(dataset_root_dir, ignore_errors=True)

    dataset_jsonl_fp = os.path.join(dataset_root_dir, "annotation.jsonl")
    os.makedirs(os.path.join(dataset_root_dir, "images"), exist_ok=True)

    for package_name in tqdm(os.listdir(mcts_dir), desc="Processing App", ncols=80):
        if not os.path.isdir(os.path.join(mcts_dir, package_name)):
            continue
        trajectory_data_dir = os.path.join(mcts_dir, package_name, "execute")
        tree_fp = os.path.join(mcts_dir, package_name, "tree.pkl.zst")
        tree_root, tree_nodes = load_object_from_disk(tree_fp)

        for trajectory_data_fn in os.listdir(trajectory_data_dir):
            trajectory_data_fp = os.path.join(trajectory_data_dir, trajectory_data_fn)
            trajectory_data = load_object_from_disk(trajectory_data_fp)
            if "recall" not in trajectory_data:
                continue
            uid = extract_uid(trajectory_data_fn)
            parent_node = tree_nodes.get(uid, MCTS_Node())
            visits = parent_node.visits
            score = parent_node.score
            md5_fn = get_md5_hash(os.path.basename(trajectory_data_fp))

            high_level_instruction = None
            if "high_level_instruction" in trajectory_data:
                high_level_instruction = trajectory_data["high_level_instruction"]
            else:
                high_level_instruction = _get_high_level_instruction_from_explore(
                    trajectory_data_fp
                )
            if high_level_instruction is None:
                continue
            execution_trajectory_data = trajectory_data["execution_trajectory_data"]
            recall = trajectory_data["recall"]
            tid = str(uuid.uuid4())

            with open(dataset_jsonl_fp, "a", encoding="utf-8") as f:
                for i, data in enumerate(execution_trajectory_data):
                    uuid_str = str(uuid.uuid4())  # Generate a unique ID

                    img_fn = f"{md5_fn}_{i}.webp"
                    img_r_fp = os.path.join("images", img_fn)
                    ndarray_screenshot = data["benchmark_screenshot"]
                    pil_screenshot = Image.fromarray(ndarray_screenshot).convert("RGB")
                    pil_screenshot = resize_pil_image(
                        pil_screenshot, target_max_size=1024
                    )  # 缩小图片尺寸来加快训练
                    pil_screenshot.save(
                        os.path.join(dataset_root_dir, img_r_fp),
                        format="webp",
                        quality=95,
                    )
                    ui_elements = [UIElement(**ele) for ele in data["ui_elements"]]
                    converted_action = data["converted_action"]
                    if isinstance(converted_action, str):
                        continue
                    target_element = data["target_element"]
                    if target_element is not None:
                        target_element = UIElement(**target_element)
                    if converted_action.index is not None and target_element is None:
                        print(
                            f"skipping {md5_fn}_{i} due to missing target element. {converted_action}"
                        )
                        continue
                    d = {
                        "recall": recall,
                        "visits": visits,
                        "score": score,
                        "package_name": package_name,
                        "id": f"{md5_fn}_{i}_type_1",
                        "image": img_r_fp,
                        "width": pil_screenshot.width,
                        "height": pil_screenshot.height,
                        "uuid": uuid_str,
                        "tid": tid,
                        "conversations": [
                            {"from": "human", "value": "<image>\nuser input"},
                            {"from": "gpt", "value": "assistant output"},
                        ],
                    }
                    hp = PROMPT_FOR_PLANNING_TRAINING_HUMAN.format(
                        high_level_instruction=high_level_instruction,
                        action_history=generate_action_history(
                            i, execution_trajectory_data
                        ),
                        a11y_tree=generate_a11y_tree(ui_elements),
                    )
                    gp = PROMPT_FOR_PLANNING_TRAINING_GPT.format(
                        low_level_thought=data["reasoning"],
                        action=generate_action_text(converted_action, target_element),
                    )
                    d["conversations"][0]["value"] = hp
                    d["conversations"][1]["value"] = gp
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")

                    d["id"] = f"{md5_fn}_{i}_type_2"
                    hp = PROMPT_FOR_ACTION_TRAINING_HUMAN.format(
                        action_history=generate_action_history(
                            i, execution_trajectory_data
                        ),
                        a11y_tree=generate_a11y_tree(ui_elements),
                        low_level_thought=data["reasoning"],
                    )
                    gp = PROMPT_FOR_ACTION_TRAINING_GPT.format(
                        action=generate_action_text(converted_action, target_element),
                    )
                    d["conversations"][0]["value"] = hp
                    d["conversations"][1]["value"] = gp
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print("Filtering using UCB1")
    filtered_dataset_jsonl_fp = os.path.join(
        dataset_root_dir, "filtered_annotation.jsonl"
    )
    package_name_to_data_id = (
        {}
    )  # key:package_name, value: list of trajectory id (uuid)
    uuid_to_data = {}  # key:uuid, value: trajectory data
    with open(dataset_jsonl_fp, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            package_name = data["package_name"]
            uuid_str = data["tid"]  # Use tid as the unique identifier
            if package_name not in package_name_to_data_id:
                package_name_to_data_id[package_name] = []
            if uuid_str not in uuid_to_data:
                uuid_to_data[uuid_str] = []
            package_name_to_data_id[package_name].append(uuid_str)
            uuid_to_data[uuid_str].append(data)

    for k in package_name_to_data_id.keys():
        package_name_to_data_id[k] = list(set(package_name_to_data_id[k]))  # 去重

    package_name_to_parent_visits = {}
    for package_name, uid_list in package_name_to_data_id.items():
        parent_visits = 0
        for uid in uid_list:
            data = uuid_to_data[uid][0]  # Get the first data entry for this UUID
            parent_visits += data["visits"]
        package_name_to_parent_visits[package_name] = parent_visits

    # Keep track of package names that still have data
    active_package_names = list(package_name_to_data_id.keys())
    out = []  # List to store selected trajectories
    while len(out) < max_sample_num and active_package_names:
        candidates = []
        packages_to_remove_next_iter = []  # Track packages exhausted in this round

        for package_name in active_package_names:
            if package_name_to_data_id[package_name]:  # Check if list is not empty
                # Get the next available trajectory for this package (first in the list)
                trajectory_data_id = package_name_to_data_id[package_name][0]
                trajectory_data = uuid_to_data[trajectory_data_id]
                # Safely get score and visits, providing defaults
                trajectory_data_0 = trajectory_data[
                    0
                ]  # Get the first data entry for this UUID
                score = trajectory_data_0.get("score", 0.0)
                visits = trajectory_data_0.get("visits", 0)
                # Ensure score and visits are numeric
                if not isinstance(score, (int, float)):
                    print(
                        f"Warning: Invalid 'score' type for {trajectory_data_0.get('id', 'N/A')}. Treating as 0.0."
                    )
                    score = 0.0
                if not isinstance(visits, (int, float)):
                    print(
                        f"Warning: Invalid 'visits' type for {trajectory_data_0.get('id', 'N/A')}. Treating as 0."
                    )
                    visits = 0
                visits = max(0, int(visits))  # Ensure visits is a non-negative integer
                parent_visits = package_name_to_parent_visits.get(package_name, 0)
                # Calculate UCT score for this candidate trajectory
                uct_score = uct(score, visits, parent_visits)
                # Store (uct_score, package_name, index_in_list=0) - index isn't strictly needed here as we always pop(0)
                candidates.append((uct_score, package_name, trajectory_data))
            else:
                # This package's list is empty, mark it for removal from active consideration
                packages_to_remove_next_iter.append(package_name)
        # Remove exhausted packages from the active list *before* selecting the best
        # This prevents trying to access empty lists again
        for pkg_name in packages_to_remove_next_iter:
            if (
                pkg_name in active_package_names
            ):  # Ensure it hasn't been removed already
                active_package_names.remove(pkg_name)
        if not candidates:
            # No more trajectories available in any active package
            print("No more candidates available.")
            break
        # Select the candidate trajectory with the highest UCT score
        # If scores are equal, max() typically takes the first one encountered.
        best_candidate = max(candidates, key=lambda item: item[0])
        best_uct_score, best_package_name, best_trajectory_data = best_candidate
        # Add the selected trajectory to the output list
        out.append(best_trajectory_data)
        # Remove the selected trajectory from its pool (the one we just took from index 0)
        package_name_to_data_id[best_package_name].pop(0)
        # Check if the package list became empty *after* popping
        if not package_name_to_data_id[best_package_name]:
            if best_package_name in active_package_names:  # Check again before removing
                active_package_names.remove(best_package_name)
    print(f"Selected {len(out)} trajectories.")
    # Write the selected trajectories to the filtered output file
    try:
        with open(filtered_dataset_jsonl_fp, "w", encoding="utf-8") as f_out:
            for selected_data in out:
                for data in selected_data:
                    f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
        print(f"Filtered dataset saved to {filtered_dataset_jsonl_fp}")
    except IOError as e:
        print(f"Error writing output file {filtered_dataset_jsonl_fp}: {e}")

    print("Resampling the dataset using recall")
    dataset_jsonl_fp = os.path.join(dataset_root_dir, "filtered_annotation.jsonl")
    resampled_dataset_jsonl_fp = os.path.join(
        dataset_root_dir, "resampled_annotation.jsonl"
    )
    resampled_dataset_json_qwen_fp = os.path.join(
        dataset_root_dir, "resampled_annotation_qwen.json"
    )
    package_name_to_data_id = {}  # key:package_name, value: list of trajectory data id
    uuid_to_data = {}  # key:uuid, value: trajectory data
    with open(dataset_jsonl_fp, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            package_name = data["package_name"]
            if package_name not in package_name_to_data_id:
                package_name_to_data_id[package_name] = []
            uuid_str = data["tid"]  # Use tid as the unique identifier
            if uuid_str not in uuid_to_data:
                uuid_to_data[uuid_str] = []
            package_name_to_data_id[package_name].append(uuid_str)
            uuid_to_data[uuid_str].append(data)

    for k in package_name_to_data_id.keys():
        package_name_to_data_id[k] = list(set(package_name_to_data_id[k]))

    package_name_to_min_recall = {}
    package_name_to_max_recall = {}
    for package_name, uid_list in package_name_to_data_id.items():
        min_recall = float("inf")
        max_recall = 0
        for uid in uid_list:
            recall = uuid_to_data[uid][0]["recall"]
            if recall > 0:
                min_recall = min(min_recall, recall)
                max_recall = max(max_recall, recall)
        if min_recall == float("inf"):
            raise ValueError(f"Package {package_name} has no non-zero recall values.")
        package_name_to_min_recall[package_name] = min_recall
        package_name_to_max_recall[package_name] = max_recall

    print(
        f"Package name to min recall: \n{json.dumps(package_name_to_min_recall, indent=4)}"
    )
    print(
        f"Package name to max recall: \n{json.dumps(package_name_to_max_recall, indent=4)}"
    )
    out = []
    for package_name, uid_list in package_name_to_data_id.items():
        min_recall = package_name_to_min_recall[package_name]
        for uid in uid_list:
            recall = uuid_to_data[uid][0]["recall"]
            recall_threshold = (
                package_name_to_min_recall.get(package_name, 0)
                + package_name_to_max_recall.get(package_name, 0.01)
            ) / 2
            if recall >= recall_threshold:
                repeat_times = (
                    recall + min_recall - 1e-9
                ) // min_recall  # Ceiling division using float arithmetic (handle potential precision issues)
                repeat_times = int(
                    max(1, repeat_times)
                )  # Ensure at least 1 repeat if recall > 0
                for i in range(repeat_times):
                    for idx in range(len(uuid_to_data[uid])):
                        _data = copy.deepcopy(uuid_to_data[uid][idx])
                        _data["id"] = f"{uuid_to_data[uid][idx]['id']}_resample_{i}"
                        out.append(_data)
    random_obj.shuffle(out)  # Shuffle the output list in place
    with open(resampled_dataset_jsonl_fp, "w", encoding="utf-8") as f:
        for data in out:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    with open(resampled_dataset_json_qwen_fp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Resampled dataset saved to {resampled_dataset_jsonl_fp}")
