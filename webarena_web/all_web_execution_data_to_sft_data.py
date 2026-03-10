try:
    from .utils_web import (
        load_object_from_disk,
        get_md5_hash,
        resize_pil_image,
        resize_ndarray_image,
        str2base32,
        base322str,
    )
    from .webarena_device import UIElement
    from .mcts_web import MCTS_Node
except ImportError:
    from webarena_web.utils_web import (
        load_object_from_disk,
        get_md5_hash,
        resize_pil_image,
        resize_ndarray_image,
        str2base32,
        base322str,
    )
    from webarena_web.webarena_device import UIElement
    from webarena_web.mcts_web import MCTS_Node

import os
import re
import json
from PIL import Image
from tqdm import tqdm
import random
import copy
import math
import shutil

random_seed = 42  # 随机种子
random_obj = random.Random(random_seed)

UCT_CONST = 1.414  # UCT常数，通常取2^0.5，表示探索和利用之间的权衡。


def uct(score: float, visits: int, parent_visits: int) -> float:
    """
    Calculate the Upper Confidence Bound for Trees (UCT) value for a given action tuple.
    """
    return score + UCT_CONST * math.sqrt(
        math.log(parent_visits) / (1 + visits)
    )


def extract_uid(s: str) -> str | None:
    """从文件名中提取出该轨迹的初始节点的父节点的uid"""
    pattern = r"\d{3}_(.*?)_execute_"
    match = re.search(pattern, s)
    if match:
        try:
            s=match.group(1).split("_click")[0].split("_long_click")[0].split("_scroll")[0].split("_input")[0]
            if '_' not in s and len(s)>0:
                return base322str(s)
            return s
        except Exception as e:
            print(f"Error extracting UID from {s}: {e}")
            return None
    return None


import uuid

if __name__ == "__main__":
    # NOTE: 这里可以根据需要修改成其他路径
    mcts_dir = "mcts_web_output"
    # NOTE: 这里可以根据需要修改成其他路径
    dataset_root_dir = "mcts_web_dataset_execute"  # 数据集的根路径
    # NOTE: 这里可以根据需要修改
    max_sample_num = 1000

    shutil.rmtree(dataset_root_dir, ignore_errors=True)

    dataset_jsonl_fp = os.path.join(dataset_root_dir, "annotation.jsonl")
    os.makedirs(os.path.join(dataset_root_dir, "images"), exist_ok=True)

    for home_url in tqdm(os.listdir(mcts_dir), desc="Processing Website", ncols=80):
        if not os.path.isdir(os.path.join(mcts_dir, home_url)):
            continue
        trajectory_data_dir = os.path.join(mcts_dir, home_url, "execute")
        tree_fp = os.path.join(mcts_dir, home_url, "tree.pkl.zst")
        tree_root, tree_nodes = load_object_from_disk(tree_fp)

        for trajectory_data_fn in os.listdir(trajectory_data_dir):
            trajectory_data_fp = os.path.join(trajectory_data_dir, trajectory_data_fn)
            trajectory_data = load_object_from_disk(trajectory_data_fp)
            if "recall" not in trajectory_data:
                continue
            uid = extract_uid(trajectory_data_fn)
            if uid not in tree_nodes:
                uid = trajectory_data.get(
                    "parent_node_uid", None
                )

            if uid not in tree_nodes:
                for data in trajectory_data["execution_trajectory_data"]:
                    if data["target_element"]:
                        target_element = data["target_element"]
                        _uid = target_element.get("uid", None)
                        if _uid in tree_nodes:
                            uid = tree_nodes[_uid].parent
                            print(
                                f"Found uid {uid} using fallback method for {trajectory_data_fn}"
                            )
                            break
            if uid not in tree_nodes:
                uid = tree_root.children[0]
                print(
                    f"Using root node's children {uid} as fallback for {trajectory_data_fn}"
                )
            parent_node = tree_nodes.get(
                uid, MCTS_Node()
            )
            visits = parent_node.visits
            score = parent_node.score
            md5_fn = get_md5_hash(os.path.basename(trajectory_data_fp))

            execution_trajectory_data = trajectory_data["execution_trajectory_data"]
            recall = trajectory_data["recall"]
            tid = str(uuid.uuid4())

            with open(dataset_jsonl_fp, "a", encoding="utf-8") as f:
                for i, data in enumerate(execution_trajectory_data):
                    uuid_str = str(uuid.uuid4())  # Generate a unique ID

                    img_fn = f"{md5_fn}_{i}.webp"
                    img_r_fp = os.path.join("images", img_fn)
                    ndarray_screenshot = data["screenshot"]
                    pil_screenshot = Image.fromarray(ndarray_screenshot).convert("RGB")
                    pil_screenshot.save(
                        os.path.join(dataset_root_dir, img_r_fp),
                        format="webp",
                        quality=95,
                    )
                    d = {
                        "recall": recall,
                        "visits": visits,
                        "score": score,
                        "home_url": home_url,
                        "id": f"{md5_fn}_{i}_type_1",
                        "image": img_r_fp,
                        "width": pil_screenshot.width,
                        "height": pil_screenshot.height,
                        "uuid": uuid_str,
                        "tid": tid,  # trajectory id
                        "conversations": [
                            {"from": "human", "value": "<image>\nuser input"},
                            {"from": "gpt", "value": "assistant output"},
                        ],
                    }
                    agent_info = data["agent_info"]
                    chat_messages = agent_info["chat_messages"]
                    messages = chat_messages.to_openai()
                    assert messages[0]["role"] == "system"
                    assert messages[1]["role"] == "user"
                    assert messages[2]["role"] == "assistant"
                    hp = (
                        "<image>\n"
                        + messages[0]["content"]
                        + "\n\n"
                        + messages[1]["content"]
                    )
                    gp = messages[2]["content"]
                    d["conversations"][0]["value"] = hp
                    d["conversations"][1]["value"] = gp
                    f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print("Filtering using UCB1")
    filtered_dataset_jsonl_fp = os.path.join(
        dataset_root_dir, "filtered_annotation.jsonl"
    )
    home_url_to_data_id = {}  # key:home_url, value: list of trajectory id (uuid)
    uuid_to_data = {}  # key:uuid, value: trajectory data
    with open(dataset_jsonl_fp, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            home_url = data["home_url"]
            uuid_str = data["tid"]  # Use tid as the unique identifier
            if home_url not in home_url_to_data_id:
                home_url_to_data_id[home_url] = []
            if uuid_str not in uuid_to_data:
                uuid_to_data[uuid_str] = []
            home_url_to_data_id[home_url].append(uuid_str)
            uuid_to_data[uuid_str].append(data)

    for k in home_url_to_data_id.keys():
        home_url_to_data_id[k] = list(set(home_url_to_data_id[k]))

    home_url_to_parent_visits = {}
    for home_url, uid_list in home_url_to_data_id.items():
        parent_visits = 0
        for uid in uid_list:
            data = uuid_to_data[uid][0]  # Get the first data entry for this UUID
            parent_visits += data["visits"]
        home_url_to_parent_visits[home_url] = parent_visits

    # Keep track of Websites that still have data
    active_home_urls = list(home_url_to_data_id.keys())
    out = []  # List to store selected trajectories
    while len(out) < max_sample_num and active_home_urls:
        candidates = []
        websites_to_remove_next_iter = []  # Track websites exhausted in this round

        for home_url in active_home_urls:
            if home_url_to_data_id[home_url]:  # Check if list is not empty
                # Get the next available trajectory for this website (first in the list)
                trajectory_data_id = home_url_to_data_id[home_url][0]
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
                parent_visits = home_url_to_parent_visits.get(home_url, 0)
                # Calculate UCT score for this candidate trajectory
                uct_score = uct(score, visits, parent_visits)
                # Store (uct_score, home_url, index_in_list=0) - index isn't strictly needed here as we always pop(0)
                candidates.append((uct_score, home_url, trajectory_data))
            else:
                # This website's list is empty, mark it for removal from active consideration
                websites_to_remove_next_iter.append(home_url)
        # Remove exhausted websites from the active list *before* selecting the best
        # This prevents trying to access empty lists again
        for home_url in websites_to_remove_next_iter:
            if home_url in active_home_urls:  # Ensure it hasn't been removed already
                active_home_urls.remove(home_url)
        if not candidates:
            # No more trajectories available in any active website
            print("No more candidates available.")
            break
        # Select the candidate trajectory with the highest UCT score
        # If scores are equal, max() typically takes the first one encountered.
        best_candidate = max(candidates, key=lambda item: item[0])
        best_uct_score, best_home_url, best_trajectory_data = best_candidate
        # Add the selected trajectory to the output list
        out.append(best_trajectory_data)
        # Remove the selected trajectory from its pool (the one we just took from index 0)
        home_url_to_data_id[best_home_url].pop(0)
        # Check if the website list became empty *after* popping
        if not home_url_to_data_id[best_home_url]:
            if best_home_url in active_home_urls:  # Check again before removing
                active_home_urls.remove(best_home_url)
    print(f"Selected {len(out)} trajectories.")
    # Write the selected trajectories to the filtered output file
    try:
        with open(filtered_dataset_jsonl_fp, "w", encoding="utf-8") as f_out:
            for selected_data in out:
                for data in selected_data:
                    f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
        print(f"Filtered dataset saved to {filtered_dataset_jsonl_fp}")
        with open(
            os.path.join(dataset_root_dir, "filtered_annotation_qwen.json"),
            "w",
            encoding="utf-8",
        ) as f:
            _out = []
            for selected_data in out:
                _out.extend(selected_data)
            json.dump(_out, f, ensure_ascii=False, indent=2)
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
    resampled_dataset_wo_img_jsonl_fp = os.path.join(
        dataset_root_dir, "resampled_annotation_wo_img.jsonl"
    )
    resampled_dataset_wo_img_json_qwen_fp = os.path.join(
        dataset_root_dir, "resampled_annotation_qwen_wo_img.json"
    )
    home_url_to_data_id = {}  # key:home_url, value: list of trajectory data id
    uuid_to_data = {}  # key:uuid, value: trajectory data
    with open(dataset_jsonl_fp, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            home_url = data["home_url"]
            if home_url not in home_url_to_data_id:
                home_url_to_data_id[home_url] = []
            uuid_str = data["tid"]  # Use tid as the unique identifier
            if uuid_str not in uuid_to_data:
                uuid_to_data[uuid_str] = []
            home_url_to_data_id[home_url].append(uuid_str)
            uuid_to_data[uuid_str].append(data)

    for k in home_url_to_data_id.keys():
        home_url_to_data_id[k] = list(set(home_url_to_data_id[k]))

    home_url_to_min_recall = {}
    home_url_to_max_recall = {}
    for home_url, uid_list in home_url_to_data_id.items():
        min_recall = float("inf")
        max_recall = 0
        for uid in uid_list:
            recall = uuid_to_data[uid][0]["recall"]
            if recall > 0:
                min_recall = min(min_recall, recall)
                max_recall = max(max_recall, recall)
        if min_recall == float("inf"):
            raise ValueError(f"Website {home_url} has no non-zero recall values.")
        home_url_to_min_recall[home_url] = min_recall
        home_url_to_max_recall[home_url] = max_recall

    print(f"Website to min recall: \n{json.dumps(home_url_to_min_recall, indent=4)}")
    print(f"Website to max recall: \n{json.dumps(home_url_to_max_recall, indent=4)}")
    out = []
    alpha = 0.3
    max_repeat_times = 5
    for home_url, uid_list in home_url_to_data_id.items():
        min_recall = home_url_to_min_recall[home_url]
        for uid in uid_list:
            recall = uuid_to_data[uid][0]["recall"]
            recall_threshold = (1 - alpha) * home_url_to_min_recall.get(
                home_url, 0
            ) + alpha * home_url_to_max_recall.get(home_url, 0.01)
            if recall >= recall_threshold:
                repeat_times = (
                    recall + min_recall - 1e-9
                ) // min_recall  # Ceiling division using float arithmetic (handle potential precision issues)
                repeat_times = int(
                    max(1, repeat_times)
                )  # Ensure at least 1 repeat if recall > 0
                repeat_times = min(
                    repeat_times, max_repeat_times
                )  # Cap the maximum repeats
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
    for i in range(len(out)):
        out[i].pop("image")
        out[i].pop("width")
        out[i].pop("height")
        out[i]["conversations"][0]["value"] = out[i]["conversations"][0][
            "value"
        ].removeprefix("<image>\n")
    with open(resampled_dataset_wo_img_jsonl_fp, "w", encoding="utf-8") as f:
        for data in out:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
    with open(resampled_dataset_wo_img_json_qwen_fp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Resampled dataset saved to {resampled_dataset_jsonl_fp}")
