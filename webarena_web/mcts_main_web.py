import os

os.environ["no_proxy"] = "localhost, 127.0.0.1/8, ::1"
import json

from .bootstrap_agent.generic_agent import GenericAgent
from .webarena_device import WebDevice
from typing import Dict
from .utils_web import (
    load_object_from_disk,
    save_object_to_disk,
    trajectory_to_instruction,
    construct_new_filepath,
    update_documents,
    get_md5_hash,
    str2base32,
    base322str,
)
from .mcts_web import (
    MCTS_Node,
    selection,
    expansion,
    simulation,
    verifier,
    get_candidate_nodes_for_selection,
    execute_mcts_action,
    refine_high_level_instruction,
    IS_VISITED_NODE,
)
from datetime import datetime
from tqdm import tqdm
import time
import hashlib


def MCTS(
    home_url: str,
    root_data_dir: str,
    device_controller: WebDevice,
    max_simulation_steps: int = 6,
    max_execution_steps: int = 30,
    max_execution_retries: int = 3,
    max_branching_factor: int = 5,
    recall_threshold: float = 0.5,
    usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0},
):
    """
    Monte Carlo Tree Search (MCTS) algorithm for exploring UI elements.
    """
    global IS_VISITED_NODE
    IS_VISITED_NODE = {}
    MCTS_ID = datetime.now().strftime("%Y%m%d%H%M%S.%f")
    print(MCTS_ID)
    device_controller.goto(home_url)
    time.sleep(2)  # 等待页面加载完成
    agent = GenericAgent(home_url=device_controller.home_url)

    data_dir = os.path.join(root_data_dir, str2base32(home_url))
    os.makedirs(data_dir, exist_ok=True)
    tree_pkl_fp = os.path.join(data_dir, "tree.pkl.zst")
    documents_pkl_fp = os.path.join(data_dir, "documents.pkl.zst")
    explore_pkl_dir = os.path.join(data_dir, "explore")
    os.makedirs(explore_pkl_dir, exist_ok=True)
    execute_pkl_dir = os.path.join(data_dir, "execute")
    os.makedirs(execute_pkl_dir, exist_ok=True)
    usage_json_dir = os.path.join(data_dir, "usage")
    os.makedirs(usage_json_dir, exist_ok=True)

    tree_root, tree_nodes = None, {}
    if os.path.exists(tree_pkl_fp):
        tree_root, tree_nodes = load_object_from_disk(tree_pkl_fp)
    documents = {}
    if os.path.exists(documents_pkl_fp):
        documents = load_object_from_disk(documents_pkl_fp)

    traceback_action = []
    parent_uid = None
    score = 0.0
    exception = None

    print("\nBegin Selection\n")
    try:
        if tree_root is None:
            tree_root = MCTS_Node(ele=None, parent=None)
            tree_root.uid = home_url
            tree_root.visits = 10
            tree_nodes[tree_root.uid] = tree_root
            parent_uid = tree_root.uid
        else:
            # 进入选择阶段
            parent_uid = tree_root.uid
            while True:
                time.sleep(3)
                selected_action = selection(
                    nodes=get_candidate_nodes_for_selection(
                        eles=device_controller.get_ui_elements(),
                        parent_uid=parent_uid,
                        tree_nodes=tree_nodes,
                    ),
                    tree_nodes=tree_nodes,
                )
                if selected_action is None:
                    break
                execute_mcts_action(
                    selected_action,
                    device_controller=device_controller,
                    tree_nodes=tree_nodes,
                    usage=usage,
                )
                traceback_action.append(selected_action)
                parent_uid = selected_action[0]
        usage_json_fp = construct_new_filepath(
            usage_json_dir, f"usage_after_selection_{MCTS_ID}.json"
        )
        with open(usage_json_fp, "w", encoding="utf-8") as f:
            json.dump(usage, f, indent=2, ensure_ascii=False)

        print("\nBegin Expansion\n")
        time.sleep(3)
        selected_action = expansion(
            eles=device_controller.get_ui_elements(),
            parent_uid=parent_uid,
            tree_nodes=tree_nodes,
            max_branching_factor=max_branching_factor,
        )
        if selected_action is None:
            print("Warning: No action found in expansion")
            save_object_to_disk((tree_root, tree_nodes), tree_pkl_fp)
            return None
        execute_mcts_action(
            selected_action,
            device_controller=device_controller,
            tree_nodes=tree_nodes,
            usage=usage,
        )
        traceback_action.append(selected_action)
        parent_uid = selected_action[0]
        usage_json_fp = construct_new_filepath(
            usage_json_dir, f"usage_after_expansion_{MCTS_ID}.json"
        )
        with open(usage_json_fp, "w", encoding="utf-8") as f:
            json.dump(usage, f, indent=2, ensure_ascii=False)

        url_after_expansion = device_controller.get_current_url()

        print("\nBegin Simulation\n")
        exploration_trajectory_data = simulation(
            device_controller=device_controller,
            tree_nodes=tree_nodes,
            max_steps=max_simulation_steps,
            usage=usage,
        )
        if len(exploration_trajectory_data["actions"]) == 0:
            print("Warning: No actions found in simulation")
            save_object_to_disk((tree_root, tree_nodes), tree_pkl_fp)
            return
        trajectory_to_instruction(exploration_trajectory_data, usage=usage)
        explore_pkl_fp = construct_new_filepath(
            explore_pkl_dir,
            f"{traceback_action[-1][1]}_{traceback_action[-1][2]}_explore_{MCTS_ID}.pkl.zst",
        )
        if len(explore_pkl_fp) > 245:
            dirname = os.path.dirname(explore_pkl_fp)
            filename = os.path.basename(explore_pkl_fp)
            ext_name = os.path.splitext(filename)[1]
            ext = ext_name if ext_name else ".pkl.zst"
            hashed_filename = hashlib.md5(filename.encode()).hexdigest()
            explore_pkl_fp = os.path.join(dirname, f"{hashed_filename}{ext}")
            print(f"File path too long, changed to {explore_pkl_fp}")
        save_object_to_disk(exploration_trajectory_data, explore_pkl_fp)
        update_documents(exploration_trajectory_data, documents)
        save_object_to_disk(documents, documents_pkl_fp)
        usage_json_fp = construct_new_filepath(
            usage_json_dir, f"usage_after_simulation_{MCTS_ID}.json"
        )
        with open(usage_json_fp, "w", encoding="utf-8") as f:
            json.dump(usage, f, indent=2, ensure_ascii=False)

        print("\nBegin Execution\n")
        instruction = exploration_trajectory_data["high_level_instruction"]
        score = 0.0
        exploration_trajectory_data["recalls"] = []
        for i in range(max_execution_retries + 1):
            device_controller.goto(url_after_expansion)
            time.sleep(3)
            execution_trajectory_data = agent.run(
                task_goal=instruction,
                max_rounds=max_execution_steps,
            )
            recall, matched_low_level_instructions, matched_gui_agent_steps = verifier(
                exploration_trajectory_data=exploration_trajectory_data,
                execution_trajectory_data=execution_trajectory_data,
                usage=usage,
            )
            exploration_trajectory_data["recalls"].append(recall)
            execution_data = {
                "execution_trajectory_data": execution_trajectory_data,
                "recall": recall,
                "matched_low_level_instructions": matched_low_level_instructions,
                "matched_gui_agent_steps": matched_gui_agent_steps,
                "is_refined": i >= 1,
                "high_level_instruction": instruction,
                "parent_node_uid": traceback_action[-1][0],
            }

            execute_pkl_fp = construct_new_filepath(
                execute_pkl_dir,
                f"{traceback_action[-1][1]}_{traceback_action[-1][2]}_execute_{MCTS_ID}.pkl.zst",
            )
            if len(execute_pkl_fp) > 245:
                dirname = os.path.dirname(execute_pkl_fp)
                filename = os.path.basename(execute_pkl_fp)
                ext_name = os.path.splitext(filename)[1]
                ext = ext_name if ext_name else ".pkl.zst"
                hashed_filename = hashlib.md5(filename.encode()).hexdigest()
                execute_pkl_fp = os.path.join(
                    dirname, f"{i+1:03d}_{hashed_filename}{ext}"
                )
                print(f"File path too long, changed to {execute_pkl_fp}")
            save_object_to_disk(execution_data, execute_pkl_fp)
            print(f"Recall: {recall}")
            if recall >= recall_threshold:
                score = 1 / recall
                break
            if i == max_execution_retries:
                print(f"Max execution retries reached: {max_execution_retries}")
                break
            refined_high_level_instruction = refine_high_level_instruction(
                high_level_instruction=instruction,
                exploration_trajectory_data=exploration_trajectory_data,
                execution_trajectory_data=execution_trajectory_data,
                matched_low_level_instructions=matched_low_level_instructions,
                matched_gui_agent_steps=matched_gui_agent_steps,
                usage=usage,
            )
            instruction = refined_high_level_instruction
            if "refined_high_level_instructions" not in exploration_trajectory_data:
                exploration_trajectory_data["refined_high_level_instructions"] = []
            exploration_trajectory_data["refined_high_level_instructions"].append(
                refined_high_level_instruction
            )
            print(f"Retry {i + 1}/{max_execution_retries} due to low recall: {recall}")
            max_execution_steps = int(1.3 * max_execution_steps)
            print(f"New max_execution_steps: {max_execution_steps} (increased by 30%)")

        save_object_to_disk(exploration_trajectory_data, explore_pkl_fp)
        usage_json_fp = construct_new_filepath(
            usage_json_dir, f"usage_after_execution_{MCTS_ID}.json"
        )
        with open(usage_json_fp, "w", encoding="utf-8") as f:
            json.dump(usage, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"An error occurred: {e}")
        exception = e
        import traceback

        traceback.print_exception(e)
        return None
    print("\nBegin Back Propagation\n")
    for selected_action in traceback_action:
        uid, action_type, action_name = selected_action
        node_visits = tree_nodes[uid].visits
        node_score = tree_nodes[uid].score
        tree_nodes[uid].score = (
            node_visits - 1
        ) * node_score / node_visits + score / node_visits
        action_visits = tree_nodes[uid].actions[action_type]["actions"][action_name][
            "visits"
        ]
        action_score = tree_nodes[uid].actions[action_type]["actions"][action_name][
            "score"
        ]
        tree_nodes[uid].actions[action_type]["actions"][action_name]["score"] = (
            action_visits - 1
        ) * action_score / action_visits + score / action_visits

    save_object_to_disk((tree_root, tree_nodes), tree_pkl_fp)
    save_object_to_disk(documents, documents_pkl_fp)
    if exception is not None:
        raise exception


import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCTS for UI exploration")
    parser.add_argument(
        "--home_url",
        type=str,
        required=True,
        help="Home url of the website to explore",
    )
    parser.add_argument(
        "--root_data_dir",
        type=str,
        default="./mcts_web_output",
        help="Root directory to save exploration data",
    )
    parser.add_argument(
        "--max_simulation_steps",
        type=int,
        default=6,
        help="Maximum steps for simulation",
    )
    parser.add_argument(
        "--max_execution_steps",
        type=int,
        default=30,
        help="Maximum steps for execution",
    )
    parser.add_argument(
        "--max_execution_retries",
        type=int,
        default=3,
        help="Maximum retries for execution",
    )
    parser.add_argument(
        "--max_branching_factor",
        type=int,
        default=5,
        help="Maximum branching factor for MCTS",
    )
    parser.add_argument(
        "--recall_threshold",
        type=float,
        default=0.5,
        help="Recall threshold for MCTS",
    )
    parser.add_argument(
        "--iteration_num",
        type=int,
        default=1,
        help="Number of iterations for MCTS",
    )
    args = parser.parse_args()
    print(f"Args: {args}")

    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    try:
        device_controller = WebDevice(home_url=args.home_url)
        print("Running MCTS...")
        for i in tqdm(range(args.iteration_num), desc="MCTS Iterations", ncols=80):
            print(f"Iteration {i + 1}/{args.iteration_num}")
            MCTS(
                home_url=args.home_url,
                root_data_dir=args.root_data_dir,
                device_controller=device_controller,
                max_simulation_steps=args.max_simulation_steps,
                max_execution_steps=args.max_execution_steps,
                max_execution_retries=args.max_execution_retries,
                max_branching_factor=args.max_branching_factor,
                recall_threshold=args.recall_threshold,
                usage=usage,
            )
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback

        traceback.print_exc()
    finally:
        pass
    print("MCTS process completed.")
    print(f"Usage: {usage}")

# python -m webarena_web.mcts_main_web --home_url http://127.0.0.1:12001/ --root_data_dir ./mcts_web_output --max_simulation_steps 6 --max_execution_steps 6 --max_execution_retries 3 --max_branching_factor 5 --recall_threshold 0.5 --iteration_num 1
