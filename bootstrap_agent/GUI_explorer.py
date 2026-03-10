import urllib3

urllib3.disable_warnings()
from typing import Any, Optional
import xml.etree.ElementTree as ET
from dataclasses import asdict
import time
from PIL import Image
from device import (
    BoundingBox,
    UIElement,
    _normalize_bounding_box,
    _parse_ui_hierarchy,
    xml_dump_to_ui_elements,
    _generate_ui_element_description,
    validate_ui_element,
    _generate_ui_elements_description_list,
    _logical_to_physical,
    _ui_element_logical_corner,
    add_ui_element_mark,
    add_screenshot_label,
    parse_reason_action_output,
    Device,
)
from utils import extract_json, openai_request
import re

import numpy as np

from typing import Any, Optional, Dict, List, Tuple


import io
import base64
from prompt_templates import REASONING, SUMMARY


def _action_selection_prompt(
    goal: str,
    history: list[str],
    ui_elements: str,
    knowledge_prompt: str = "",
) -> str:
    """Generate the prompt for the action selection.

    Args:
      goal: The current goal.
      history: Summaries for previous steps.
      ui_elements: A list of descriptions for the UI elements.

    Returns:
      The text prompt for action selection that will be sent to gpt4v.
    """
    if history:
        history = "\n".join(history)
    else:
        history = "You just started, no action has been performed yet."

    return REASONING.format(
        task_goal=goal,
        history=history,
        ui_elements=ui_elements if ui_elements else "Not available",
        knowledge=knowledge_prompt if knowledge_prompt else "Not available",
    )


def _summarize_prompt(
    action: str,
    reasoning: str,
    goal: str,
    before_elements: str,
    after_elements: str,
) -> str:
    """Generate the prompt for the summarization step.

    Args:
      action: Action picked.
      reasoning: The reasoning to pick the action.
      goal: The overall goal.
      before_elements: Information for UI elements on the before screenshot.
      after_elements: Information for UI elements on the after screenshot.

    Returns:
      The text prompt for summarization that will be sent to gpt4v.
    """
    return SUMMARY.format(
        task_goal=goal,
        before_ui_elements=before_elements,
        after_ui_elements=after_elements,
        action=action,
        reasoning=reasoning,
    )


import bootstrap_agent.json_action as json_action

import io
import numpy as np

from utils import np_array_to_jpeg_base64, resize_ndarray_image


import requests
import os


def ask_mllm(
    text_prompt: str,
    images: list[np.ndarray],
    usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0},
) -> tuple[str, Any]:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f'Bearer {os.getenv("OPENAI_API_KEY")}',
    }

    payload = {
        "model": os.getenv("OPENAI_API_MODEL"),
        "temperature": 0.0,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                ],
            }
        ],
        "max_tokens": 1000,
    }

    for image in images:
        payload["messages"][0]["content"].append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{np_array_to_jpeg_base64(resize_ndarray_image(image,target_max_size=1000))}"
                },
            }
        )

    counter = 5  # max_retry
    wait_seconds = 1
    HTTP__PROXY = os.getenv("HTTP__PROXY", None)
    proxies = None
    if HTTP__PROXY:
        proxies = {
            "http": HTTP__PROXY,
            "https": HTTP__PROXY,
        }
    response = None
    while counter > 0:
        try:
            response = requests.post(
                os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
                + "/chat/completions",
                headers=headers,
                json=payload,
                timeout=300,
                verify=False,
                proxies=proxies,
            )
            if response.ok and "choices" in response.json():
                d = response.json()
                usage["prompt_tokens"] += d["usage"]["prompt_tokens"]
                usage["completion_tokens"] += d["usage"]["completion_tokens"]
                return response.json()["choices"][0]["message"]["content"], response
            print(
                "Error calling OpenAI API with error message: "
                + response.json()["error"]["message"]
            )
            time.sleep(wait_seconds)
            wait_seconds *= 2
        except Exception as e:  # pylint: disable=broad-exception-caught
            time.sleep(wait_seconds)
            wait_seconds *= 2
            counter -= 1
            print("Error calling LLM, will retry soon...")
            print(e)
            if response is not None:
                print(response.text)
    raise RuntimeError("Error calling LLM, please check the logs for more details.")


import copy


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
        # env=self._base_env,
        device_controller=device_controller,
        extras={"task_type_string": header, "goal_string": message},
    )


import logging


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
            # First focus on enter text UI element.
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


from utils import save_object_to_disk, get_md5_hash
from datetime import datetime
import copy


class GUI_explorer(object):
    def __init__(self, device_serial: str = None, step_interval: float = 2.0):
        self.history = []
        self.device = Device(device_serial=device_serial)
        self.step_interval = step_interval  # Wait a few seconds for the screen to stabilize after executing an action.

    def reset(self, go_home_on_reset: bool = False):
        # Hide the coordinates on screen which might affect the vision model.
        if go_home_on_reset:
            self.device.home()
        self.history = []

    def run(
        self,
        task_goal: str,
        documents: Dict[str, str] = {},
        max_rounds: int = 30,
        step_interval: float = 2.0,
        usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0},
    ) -> List[dict[str, Any]]:
        self.reset()
        print(f"Running task: {task_goal}")
        self.step_interval = step_interval
        step_datas = []
        for i in range(max_rounds):
            self.device.wait_to_stabilize()
            print(f"\nRound {i + 1}/{max_rounds}")
            stop, step_data = self.step(task_goal, documents=documents, usage=usage)
            step_datas.append(step_data)
            if stop:
                break
        print("Done.")
        return step_datas

    def step(
        self,
        goal: str,
        documents: Dict[str, str] = {},
        usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0},
    ) -> tuple[bool, dict[str, Any]]:
        step_data = {
            "raw_screenshot": None,
            "before_screenshot_with_som": None,
            "after_screenshot_with_som": None,
            "reasoning": "None",
            "action_prompt": "None",
            "action_output": "None",
            "action_raw_response": None,
            "summary_prompt": "None",
            "summary": "None",
            "summary_raw_response": None,
            "converted_action": "error_retry",
            "actual_action_coordinates": None,
            "benchmark_screenshot": None,
            "ui_elements": None,
            "top_app_package_name": None,
            "target_element": None,
        }
        print("----------step " + str(len(self.history) + 1))

        before_ui_elements = self.device.wait_to_stabilize()
        orientation = self.device.get_orientation()
        logical_screen_size = self.device.get_screen_size()
        physical_frame_boundary = self.device.get_physical_frame_boundary()

        step_data["ui_elements"] = [
            asdict(ui_element) for ui_element in before_ui_elements
        ]
        before_ui_elements_list = _generate_ui_elements_description_list(
            before_ui_elements, logical_screen_size
        )
        before_screenshot = np.array(self.device.get_screenshot())
        step_data["raw_screenshot"] = before_screenshot.copy()
        step_data["benchmark_screenshot"] = copy.deepcopy(before_screenshot)
        knowledge_prompt = ""
        all_knowledge = []
        top_app_package_name = self.device.get_top_package_name()
        step_data["top_app_package_name"] = top_app_package_name
        for index, ui_element in enumerate(before_ui_elements):
            if validate_ui_element(ui_element, logical_screen_size):
                add_ui_element_mark(
                    before_screenshot,
                    ui_element,
                    index,
                    logical_screen_size,
                    physical_frame_boundary,
                    orientation,
                )
                if ui_element.uid:
                    if ui_element.uid in documents:
                        knowledge = documents[ui_element.uid]
                        assert isinstance(
                            knowledge, str
                        ), f"Knowledge should be a string, but got {type(knowledge)}"
                        all_knowledge.append(
                            {
                                "index": index,
                                "tips": {"tip_1": knowledge},
                            }
                        )

        step_data["before_screenshot_with_som"] = before_screenshot.copy()

        prioritized_knowledge = all_knowledge
        for item in prioritized_knowledge:
            knowledge_prompt += f'\nUI element {item["index"]}: {item["tips"]['tip_1'] if len(item["tips"]) ==1 else item["tips"]}'

        if len(knowledge_prompt) > 0:
            knowledge_prompt = f"\nHere are some tips for you:{knowledge_prompt}\n"
        action_prompt = _action_selection_prompt(
            goal,
            [
                "Step " + str(i + 1) + "- " + step_info["summary"]
                for i, step_info in enumerate(self.history)
            ],
            before_ui_elements_list,
            knowledge_prompt=knowledge_prompt,
        )
        step_data["action_prompt"] = action_prompt
        action_output, raw_response = ask_mllm(
            action_prompt,
            [
                step_data["raw_screenshot"],
                before_screenshot,
            ],
            usage=usage,
        )

        if not raw_response:
            raise RuntimeError("Error calling LLM in action selection phase.")
        step_data["action_output"] = action_output.strip()
        step_data["action_raw_response"] = raw_response

        reason, action = parse_reason_action_output(action_output)

        # If the output is not in the right format, add it to step summary which
        # will be passed to next step and return.
        if (not reason) or (not action):
            print("Action prompt output is not in the correct format.")
            step_data["summary"] = (
                "Output for action selection is not in the correct format, so no"
                " action is performed."
            )
            self.history.append(step_data)

            return (False, step_data)

        action=action.strip()
        reason=reason.strip()
        print("Reasoning: " + reason)
        print("Action: " + action)
        
        step_data["reasoning"] = reason

        try:
            converted_action = json_action.JSONAction(
                **extract_json(action),
            )
            step_data["converted_action"] = converted_action
        except Exception as e:  # pylint: disable=broad-exception-caught
            print("Failed to convert the output to a valid action.")
            print(str(e))
            step_data["summary"] = (
                "Can not parse the output to a valid action. Please make sure to pick"
                " the action from the list with required parameters (if any) in the"
                " correct JSON format!"
            )
            self.history.append(step_data)
            step_data["converted_action"] = "error_retry"
            return (False, step_data)

        if (
            converted_action.action_type
            in ["click", "long_press", "input_text", "scroll"]
            and converted_action.index is not None
        ):
            if converted_action.index >= len(before_ui_elements):
                print("Index out of range.")
                step_data["summary"] = (
                    "The parameter index is out of range. Remember the index must be in"
                    " the UI element list!"
                )
                self.history.append(step_data)
                step_data["converted_action"] = "error_retry"
                return (False, step_data)

            # Add mark to the target element.
            add_ui_element_mark(
                step_data["raw_screenshot"],
                before_ui_elements[converted_action.index],
                converted_action.index,
                logical_screen_size,
                physical_frame_boundary,
                orientation,
            )
            step_data["target_element"] = asdict(
                before_ui_elements[converted_action.index]
            )

        if converted_action.action_type == "status":
            step_data["summary"] = "Agent thinks the request has been completed."
            if converted_action.goal_status == "infeasible":
                print("Agent stopped since it thinks mission impossible.")
                step_data["summary"] = (
                    "Agent thinks the mission is infeasible and stopped."
                )
            else:
                print("Agent thinks the request has been completed.")
            self.history.append(step_data)
            return (True, step_data)  # complete和infeasible都返回True，表示任务结束

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
            step_data["summary"] = (
                "Can not execute the action, make sure to select the action with"
                " the required parameters (if any) in the correct JSON format!"
            )
            step_data["converted_action"] = "error_retry"
            return (False, step_data)

        self.device.wait_to_stabilize()

        orientation = self.device.get_orientation()
        logical_screen_size = self.device.get_screen_size()
        physical_frame_boundary = self.device.get_physical_frame_boundary()

        after_ui_elements = self.device._get_ui_elements()
        after_ui_elements_list = _generate_ui_elements_description_list(
            after_ui_elements, logical_screen_size
        )
        after_screenshot = np.array(self.device.get_screenshot())
        for index, ui_element in enumerate(after_ui_elements):
            if validate_ui_element(ui_element, logical_screen_size):
                add_ui_element_mark(
                    after_screenshot,
                    ui_element,
                    index,
                    logical_screen_size,
                    physical_frame_boundary,
                    orientation,
                )

        add_screenshot_label(step_data["before_screenshot_with_som"], "before")
        add_screenshot_label(after_screenshot, "after")
        step_data["after_screenshot_with_som"] = after_screenshot.copy()

        summary_prompt = _summarize_prompt(
            action,
            reason,
            goal,
            before_ui_elements_list,
            after_ui_elements_list,
        )
        summary, raw_response = ask_mllm(
            summary_prompt,
            [
                before_screenshot,
                after_screenshot,
            ],
            usage=usage,
        )

        if not raw_response:
            step_data["summary"] = (
                "Some error occurred calling LLM during summarization phase."
            )
            self.history.append(step_data)
            return (False, step_data)

        summary=summary.strip()
        step_data["summary_prompt"] = summary_prompt
        step_data["summary"] = f"Action selected: {action}. {summary}"
        print("Summary: " + summary)
        step_data["summary_raw_response"] = raw_response

        self.history.append(step_data)
        return (False, step_data)
