from dotenv import load_dotenv
import os

dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path=dotenv_path, verbose=True, override=True)
    print("Environment loaded")

# 启动的模拟器数量
num = 20
session_name = "avds"  # The desired tmux session name

init_console_port = 5554
init_grpc_port = 8554

base_emulator_command = "unset http_proxy && unset https_proxy && unset ftp_proxy && unset all_proxy && emulator -avd AndroidWorldAvd -no-audio -no-boot-anim -feature -Vulkan -no-snapshot -no-snapshot-save -read-only -no-window"

http_proxy = os.getenv("HTTP__PROXY", None)
if http_proxy:
    # Ensure proxy string is properly quoted if it contains special characters
    # Simple check, might need more robust quoting depending on proxy format
    if any(c in http_proxy for c in " ;&|<>"):
        base_emulator_command += f' -http-proxy "{http_proxy}"'
    else:
        base_emulator_command += f" -http-proxy {http_proxy}"


# --- Generate run_avd.sh ---
output_script_path = "run_avd.sh"

with open(output_script_path, "w", encoding="utf-8") as f:
    f.write("#!/bin/bash\n\n")
    f.write(
        f"# Script to start {num} AVD instances in a tmux session named '{session_name}'\n"
    )
    f.write("# Each AVD will run in its own tmux window.\n\n")

    # Ensure tmux is installed (basic check)
    f.write("if ! command -v tmux &> /dev/null\n")
    f.write("then\n")
    f.write('    echo "Error: tmux is not installed. Please install tmux first."\n')
    f.write("    exit 1\n")
    f.write("fi\n\n")

    # Optional: Kill existing session for a clean start
    f.write(
        f"echo \"Ensuring tmux session '{session_name}' is not already running...\"\n"
    )
    # The `|| true` prevents the script from exiting if the session doesn't exist
    f.write(f'tmux kill-session -t "{session_name}" 2>/dev/null || true\n')
    # Add a small delay after killing, just in case
    f.write("sleep 0.5\n\n")

    f.write(
        f"echo \"Starting {num} AVD instances in tmux session '{session_name}'...\"\n\n"
    )

    for i in range(num):
        console_port = init_console_port + i * 2
        grpc_port = init_grpc_port + i * 2
        window_name = f"avd_{i}"  # Name windows like avd_0, avd_1, ...

        # Construct the full emulator command for this instance
        # Added quotes around the command passed to tmux for robustness
        instance_command = (
            f"{base_emulator_command} -port {console_port} -grpc {grpc_port}"
        )

        # Escape single quotes within the command if necessary (unlikely here)
        # instance_command_escaped = instance_command.replace("'", "'\\''")
        # For this specific command, direct quoting is likely fine

        f.write(f"# --- Start AVD {i} ---\n")
        f.write(
            f"echo \"Configuring window '{window_name}' (Console: {console_port}, gRPC: {grpc_port})\"\n"
        )

        if i == 0:
            # First instance: Create the new detached session
            # Pass the full command string to be executed by the shell within tmux
            tmux_command = f'tmux new-session -d -s "{session_name}" -n "{window_name}" "{instance_command}"'
        else:
            # Subsequent instances: Create new windows in the existing session
            tmux_command = f'tmux new-window -t "{session_name}" -n "{window_name}" "{instance_command}"'

        f.write(tmux_command + "\n")
        # Add a very small delay between launching tmux commands, can be adjusted or removed
        f.write("sleep 0.2\n\n")

    f.write(
        f"echo \"\nTmux session '{session_name}' setup initiated with {num} windows.\"\n"
    )
    f.write("# The emulators are starting in the background within tmux.\n")
    f.write("# It might take some time for them to fully boot.\n")
    f.write(f"# Attach to the session using: tmux attach-session -t {session_name}\n")
    f.write(f"# Or list sessions: tmux ls\n")
    f.write(
        f"# Switch windows within tmux using Ctrl+b, n (next) or Ctrl+b, p (previous) or Ctrl+b, <window_number>\n"
    )
    f.write("exit 0\n")

# --- Set Execute Permissions ---
try:
    os.chmod(output_script_path, 0o755)  # Set rwxr-xr-x permissions
    print(f"\n'{output_script_path}' created and set to executable.")
    print(
        f"Run './run_avd.sh' to start the emulators in tmux session '{session_name}'."
    )
    print(f"Then attach using 'tmux attach -t {session_name}'.")
except Exception as e:
    print(
        f"\n'{output_script_path}' created, but failed to set executable permissions: {e}"
    )
    print(f"You may need to run 'chmod +x {output_script_path}' manually.")
