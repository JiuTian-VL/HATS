import subprocess
import os
import time


def get_available_devices() -> list[str]:
    """
    Get a list of device serials connected via adb
    :return: list of str, each str is a device serial number
    """
    r = subprocess.check_output(["adb", "devices"])
    if not isinstance(r, str):
        r = r.decode()
    devices = []
    for line in r.splitlines():
        segs = line.strip().split()
        if len(segs) == 2 and segs[1] == "device":
            devices.append(segs[0])
    return devices


def stop_emulator(console_port: int, wait_until_stopped: bool = True) -> None:
    print(f"Stopping emulator with console port {console_port}...")
    device_serial = f"emulator-{console_port}"
    subprocess.run(["adb", "-s", device_serial, "emu", "kill"])
    if wait_until_stopped:
        while True:
            devices = get_available_devices()
            if device_serial not in devices:
                print(f"Emulator with console port {console_port} stopped.")
                return None
            time.sleep(1)


def wait_emulator_ready(console_port: int, timeout: float = 300) -> None:
    print(f"Waiting for emulator with console port {console_port} to be ready...")
    device_serial = f"emulator-{console_port}"
    start_time = time.time()
    while True:
        try:
            output = subprocess.check_output(
                [
                    "adb",
                    "-s",
                    device_serial,
                    "shell",
                    "getprop",
                    "sys.boot_completed",
                ],
                text=True,
                timeout=5,
            )
            if output.strip() == "1":
                print(f"Emulator with console port {console_port} is ready.")
                return None
        except subprocess.CalledProcessError:
            pass
        except Exception as e:
            print(f"Error checking emulator status: {e}")
            import traceback
            traceback.print_exc()
            raise e
        if time.time() - start_time > timeout:
            print(
                f"Timeout waiting for emulator with console port {console_port} to be ready."
            )
            raise TimeoutError(
                f"Emulator with console port {console_port} did not start in {timeout} seconds."
            )
        time.sleep(2)


def setup_emulator(
    console_port: int,
    grpc_port: int,
    emulator_name: str = "AndroidWorldAvd",
    is_multiple_env: bool = True,
    snapshot_name: str | None = None,
    no_window: bool = True,
    wait_until_ready: bool = True,
    timeout: float = 300,
) -> None:
    assert console_port % 2 == 0, "Console port must be even."
    assert (
        console_port >= 5554 and console_port <= 5682
    ), "Invalid console port, console port must in [5554, 5682]."
    command = [
        "emulator",
        "-avd",
        emulator_name,
        "-no-audio",
        "-no-boot-anim",
        "-feature",
        "-Vulkan",
        "-no-snapshot",
        "-no-snapshot-save",
        "-grpc",
        str(grpc_port),
        "-port",
        str(console_port),
        "-no-window" if no_window else "",
        "-read-only" if is_multiple_env else "",
    ]
    if snapshot_name:
        command.extend(["-snapshot", snapshot_name])
    HTTP__PROXY = os.environ.get("HTTP__PROXY", None)
    if HTTP__PROXY:
        command.extend(["-http-proxy", HTTP__PROXY])
    print(command)
    subprocess.Popen(
        command,
        text=True,
    )
    if wait_until_ready:
        wait_emulator_ready(console_port, timeout=timeout)
