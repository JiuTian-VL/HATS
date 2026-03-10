#!/usr/bin/env python3
import os
import stat  # For chmod constants
import shlex  # To safely quote shell arguments
import time
import copy  # To avoid modifying the base dictionary

# --- Configuration ---

# List of website names to process, one per tmux window
WEBSITE_NAMES = [
    "http://127.0.0.1:12001/",  # SHOPPING
    "http://127.0.0.1:12002/admin",  # SHOPPING_ADMIN, NEED LOGIN
    "http://127.0.0.1:12003",  # REDDIT
    "http://127.0.0.1:12004",  # GITLAB, NEED LOGIN
    "http://127.0.0.1:12005/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing",  # WIKIPEDIA
    "http://127.0.0.1:12006",  # MAP
    "http://127.0.0.1:12007",  # HOMEPAGE
]


# --- Script and Environment Settings ---
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TARGET_MCTS_SCRIPT_NAME = "-m webarena_web.mcts_main_web"  # Make sure this is correct
CONDA_ENV_NAME = "mcts"  # Ensure this matches your environment
PYTHON_EXECUTABLE = "python"

# --- Default MCTS Parameters (used as fallback) ---
DEFAULT_MCTS_PARAMS = {
    "MAX_SIMULATION_STEPS": 15,
    "MAX_EXECUTION_STEPS": 20,
    "MAX_EXECUTION_RETRIES": 3,
    "MAX_BRANCHING_FACTOR": 30,
    "RECALL_THRESHOLD": 0.5,
    "ITERATION_NUM": 40,
}

# --- Per-App MCTS Parameter Overrides ---
# Keys are website names, values are dictionaries of parameters to override.
# If a website is not listed, or a parameter is not specified for a listed website,
# the default value from DEFAULT_MCTS_PARAMS will be used.
APP_SPECIFIC_MCTS_PARAMS = {
    "http://127.0.0.1:12001/": {
        "MAX_SIMULATION_STEPS": 20,
        "MAX_EXECUTION_STEPS": 25,
    },
    "http://127.0.0.1:12002/admin": {  # SHOPPING_ADMIN, NEED LOGIN
        "MAX_SIMULATION_STEPS": 20,
        "MAX_EXECUTION_STEPS": 25,
        "ITERATION_NUM": max(int(DEFAULT_MCTS_PARAMS["ITERATION_NUM"] / 3), 1),
    },
    "http://127.0.0.1:12003": {
        "MAX_SIMULATION_STEPS": 20,
        "MAX_EXECUTION_STEPS": 25,
    },
    "http://127.0.0.1:12004": {  # GITLAB, NEED LOGIN
        "MAX_SIMULATION_STEPS": 20,
        "MAX_EXECUTION_STEPS": 25,
        "ITERATION_NUM": max(int(DEFAULT_MCTS_PARAMS["ITERATION_NUM"] / 3), 1),
    },
    "http://127.0.0.1:12005/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing": {
        "MAX_SIMULATION_STEPS": 20,
        "MAX_EXECUTION_STEPS": 25,
    },
    "http://127.0.0.1:12006": {
        "MAX_SIMULATION_STEPS": 20,
        "MAX_EXECUTION_STEPS": 25,
    },
    "http://127.0.0.1:12007": {
        "MAX_SIMULATION_STEPS": 20,
        "MAX_EXECUTION_STEPS": 25,
    },
}


# --- Tmux Configuration ---
SESSION_NAME = "mcts"
OUTPUT_SCRIPT_PATH = os.path.join(SCRIPT_DIR, "run_web_mcts_batch.sh")
ROOT_DATA_DIR = "./mcts_web_output"  # Base directory for output
# Delay between creating/sending keys to new windows (adjust if needed)
SEND_KEYS_DELAY_SECONDS = 1.0
# Extra delay specifically after creating a window before sending keys
POST_WINDOW_CREATION_DELAY = 0.5  # Added delay

# --- Script Generation ---


def get_mcts_params_for_app(website_name):
    """Retrieves MCTS parameters for a given app, using defaults as fallback."""
    # Start with a copy of the defaults
    params = copy.deepcopy(DEFAULT_MCTS_PARAMS)
    # Get app-specific overrides, or an empty dict if none exist
    overrides = APP_SPECIFIC_MCTS_PARAMS.get(website_name, {})
    # Update the defaults with any overrides found
    params.update(overrides)
    return params


def create_batch_script_sendkeys_v2():
    """Generates the tmux batch script content using send-keys (v2) with per-app params."""
    script_content = []
    script_content.append("#!/bin/bash")
    script_content.append(
        "\n# Auto-generated script to run MCTS simulations using tmux send-keys (v2)"
    )
    script_content.append(f"# Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    script_content.append(f"# Session Name: {SESSION_NAME}")
    script_content.append(f"# Target Directory: {SCRIPT_DIR}")
    script_content.append(f"# Conda Environment: {CONDA_ENV_NAME}")
    script_content.append(f"# Number of jobs: {len(WEBSITE_NAMES)}")
    script_content.append("# Uses per-app MCTS parameters where defined.")
    script_content.append("# Windows remain open after script finishes.")
    script_content.append("")

    script_content.append(f"SESSION_NAME={shlex.quote(SESSION_NAME)}")

    # --- Checks ---
    script_content.append("# --- Sanity Checks ---")
    script_content.append("if ! command -v tmux &> /dev/null; then")
    script_content.append(
        '    echo "Error: tmux is not installed. Please install tmux first."'
    )
    script_content.append("    exit 1")
    script_content.append("fi\n")
    script_content.append("if ! command -v conda &> /dev/null; then")
    script_content.append(
        '    echo "Warning: conda command not found. Conda activation might fail."'
        '    echo "         Make sure conda is initialized for your shell (e.g., run conda init bash)."'
    )
    script_content.append("fi\n")

    # --- Session Creation ---
    script_content.append("# --- Tmux Session Setup ---")
    script_content.append(f"# Ensure session '{SESSION_NAME}' is clean")
    script_content.append(f"tmux kill-session -t $SESSION_NAME 2>/dev/null || true")
    script_content.append(f"sleep 0.5")  # Give kill command time
    script_content.append(f"echo \"Creating new tmux session '$SESSION_NAME'...\"")
    # Create session detached, it will have one default window at index 0
    script_content.append(f"tmux new-session -d -s $SESSION_NAME")
    script_content.append(f"sleep 0.5")  # Allow session to stabilize

    script_content.append(
        f"\necho \"Configuring {len(WEBSITE_NAMES)} windows in tmux session '$SESSION_NAME' via send-keys...\"\n"
    )

    # Shared setup commands
    absolute_root_data_dir = os.path.abspath(os.path.join(SCRIPT_DIR, ROOT_DATA_DIR))
    cd_command = f"cd {shlex.quote(SCRIPT_DIR)}"
    conda_command = f"conda activate {shlex.quote(CONDA_ENV_NAME)}"
    proxy_unset_command = (
        "unset http_proxy && unset https_proxy && unset ftp_proxy && unset all_proxy"
    )

    # Create commands for each website name
    for i, website_name in enumerate(WEBSITE_NAMES):
        window_name = f"mcts_{i}_{website_name.split(':')[-1].split('/')[0]}"  # More descriptive name
        # Target specification: Use index for reliability after creation/rename
        target_pane = f"$SESSION_NAME:{i}"  # Target window by its index

        # --- Get MCTS parameters for this specific app ---
        mcts_params = get_mcts_params_for_app(website_name)

        # Construct the MCTS parameter arguments string
        # Note: Parameter names here MUST match the argparse names in mcts_main.py
        mcts_command_args = (
            f"--root_data_dir {shlex.quote(absolute_root_data_dir)} "  # Root dir is likely shared, adjust if needed per app
            f"--max_simulation_steps {mcts_params['MAX_SIMULATION_STEPS']} "
            f"--max_execution_steps {mcts_params['MAX_EXECUTION_STEPS']} "
            f"--max_execution_retries {mcts_params['MAX_EXECUTION_RETRIES']} "
            f"--max_branching_factor {mcts_params['MAX_BRANCHING_FACTOR']} "
            f"--recall_threshold {mcts_params['RECALL_THRESHOLD']} "
            f"--iteration_num {mcts_params['ITERATION_NUM']} "
        )

        # Construct the specific python execution command
        python_run_command = (
            f"{PYTHON_EXECUTABLE} {TARGET_MCTS_SCRIPT_NAME} "
            f"--home_url {shlex.quote(website_name)} "
            f"{mcts_command_args.strip()}"  # Use the dynamically generated args
        )

        script_content.append(f"# --- Window {i}: {website_name} ---")
        script_content.append(
            f"# MCTS Params: sim={mcts_params['MAX_SIMULATION_STEPS']}, exec={mcts_params['MAX_EXECUTION_STEPS']}, retry={mcts_params['MAX_EXECUTION_RETRIES']}, branch={mcts_params['MAX_BRANCHING_FACTOR']}, recall={mcts_params['RECALL_THRESHOLD']}, iter={mcts_params['ITERATION_NUM']}"
        )

        if i == 0:
            # First window (index 0) already exists, just rename it
            script_content.append(
                f"echo \"Renaming window 0 to '{window_name}' and configuring...\""
            )
            script_content.append(
                f"tmux rename-window -t $SESSION_NAME:0 {shlex.quote(window_name)}"
            )
            script_content.append(
                f"sleep {POST_WINDOW_CREATION_DELAY}"
            )  # Allow rename to settle
        else:
            # Create subsequent windows
            script_content.append(
                f"sleep {SEND_KEYS_DELAY_SECONDS}"
            )  # Delay before creating window
            script_content.append(
                f"echo \"Creating and configuring window '{window_name}' (index {i})...\""
            )
            script_content.append(
                f"tmux new-window -t $SESSION_NAME:{i} -n {shlex.quote(window_name)}"
            )  # Create at specific index
            script_content.append(
                f"sleep {POST_WINDOW_CREATION_DELAY}"
            )  # Delay after creating window

        # Send commands using send-keys to the target index
        script_content.append(f"# Sending commands to {target_pane}")
        script_content.append(
            f"tmux send-keys -t {target_pane} {shlex.quote(proxy_unset_command)} C-m"
        )
        script_content.append(
            f"tmux send-keys -t {target_pane} {shlex.quote(cd_command)} C-m"
        )
        script_content.append(
            f"tmux send-keys -t {target_pane} {shlex.quote(conda_command)} C-m"
        )
        script_content.append(
            f"tmux send-keys -t {target_pane} {shlex.quote(python_run_command)} C-m"
        )
        script_content.append("")  # Newline for readability

    # --- Concluding Instructions ---
    script_content.append(
        f"\n# Finished sending commands to all {len(WEBSITE_NAMES)} windows."
    )
    script_content.append(f"# Attach to the session to view progress:")
    script_content.append(f"# tmux attach-session -t $SESSION_NAME")
    script_content.append(
        f"echo \"\nTmux session '$SESSION_NAME' setup initiated with {len(WEBSITE_NAMES)} windows via send-keys.\""
    )
    script_content.append('echo "Commands are being executed in each window."')
    script_content.append(f'echo "Attach using: tmux attach -t $SESSION_NAME"')
    script_content.append("exit 0")  # Exit the setup script

    return "\n".join(script_content)


def main():
    """Generates the script file and sets permissions."""
    global TARGET_MCTS_SCRIPT_NAME
    # Ensure script name is just the basename
    if os.path.sep in TARGET_MCTS_SCRIPT_NAME or "/" in TARGET_MCTS_SCRIPT_NAME:
        TARGET_MCTS_SCRIPT_NAME = os.path.basename(TARGET_MCTS_SCRIPT_NAME)
        print(
            f"Warning: Corrected TARGET_MCTS_SCRIPT_NAME to '{TARGET_MCTS_SCRIPT_NAME}'"
        )
    if not TARGET_MCTS_SCRIPT_NAME:
        print(
            f"Warning: TARGET_MCTS_SCRIPT_NAME was empty, defaulting to 'mcts_main.py'"
        )
        TARGET_MCTS_SCRIPT_NAME = "mcts_main.py"

    script_code = create_batch_script_sendkeys_v2()  # Call the updated function

    try:
        with open(OUTPUT_SCRIPT_PATH, "w", encoding="utf-8") as f:
            f.write(script_code)
        print(f"Successfully generated '{OUTPUT_SCRIPT_PATH}'")

        # Set permissions: rwxr-xr-x (user:rwx, group:rx, other:rx)
        os.chmod(
            OUTPUT_SCRIPT_PATH,
            stat.S_IRWXU | stat.S_IRGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IXOTH,
        )
        print(f"Set '{OUTPUT_SCRIPT_PATH}' as executable.")
        print("\nTo run the MCTS batch:")
        print(f"  1. Ensure conda environment '{CONDA_ENV_NAME}' exists and is set up.")
        print(
            f"  2. Ensure target script '{TARGET_MCTS_SCRIPT_NAME}' exists in '{SCRIPT_DIR}'."
        )
        print(
            f"  3. Execute the generated script: ./{os.path.basename(OUTPUT_SCRIPT_PATH)}"
        )
        print(
            f"     (This script will create/configure the tmux session and then exit.)"
        )
        print(
            f"  4. Attach to the tmux session: tmux attach -t {shlex.quote(SESSION_NAME)}"
        )
        print(f"  5. Each window will run MCTS with potentially different parameters.")
        print(
            f"  6. Windows will remain open showing script output/shell prompt after completion."
        )

    except IOError as e:
        print(f"Error: Failed to write script file '{OUTPUT_SCRIPT_PATH}': {e}")
    except OSError as e:
        print(
            f"Error: Failed to set execute permissions for '{OUTPUT_SCRIPT_PATH}': {e}"
        )
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
