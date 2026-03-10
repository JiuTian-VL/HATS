import warnings
import logging
from beartype.roar import BeartypeDecorHintPep585DeprecationWarning

warnings.filterwarnings("ignore", category=BeartypeDecorHintPep585DeprecationWarning)
logging.getLogger("browsergym.core.env").setLevel(logging.ERROR)

import logging
from typing import List, Tuple, Any, Dict, Optional
from dataclasses import dataclass
import ast
import gymnasium as gym
from PIL import Image

# Register BrowserGym environments
import browsergym.webarena
from browsergym.core.action.highlevel import HighLevelActionSet

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import os

# --- 设置 WebArena 所需的环境变量 ---
# 请确保这些 URL 和端口与您启动 WebArena Docker 容器时的配置相匹配
os.environ["BASE_URL"] = "http://127.0.0.1"
os.environ["WA_SHOPPING"] = f"{os.environ['BASE_URL']}:12001/"
os.environ["WA_SHOPPING_ADMIN"] = f"{os.environ['BASE_URL']}:12002/admin"
os.environ["WA_REDDIT"] = f"{os.environ['BASE_URL']}:12003"
os.environ["WA_GITLAB"] = f"{os.environ['BASE_URL']}:12004"
os.environ["WA_WIKIPEDIA"] = (
    f"{os.environ['BASE_URL']}:12005/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
)
os.environ["WA_MAP"] = f"{os.environ['BASE_URL']}:12006"
os.environ["WA_HOMEPAGE"] = f"{os.environ['BASE_URL']}:12007"
os.environ["WA_FULL_RESET"] = f"{os.environ['BASE_URL']}:12008"

# WebArena 的评估器需要 OpenAI API，即使只是一个空值
os.environ["OPENAI_API_KEY_WA"] = "DUMMY_KEY"
os.environ["OPENAI_BASE_URL_WA"] = "https://api.openai.com/v1"


@dataclass
class BoundingBox:
    """Class for representing a bounding box in pixels."""

    x_min: int | float
    x_max: int | float
    y_min: int | float
    y_max: int | float

    @property
    def center(self) -> tuple[float, float]:
        """Gets center of bounding box."""
        return (self.x_min + self.x_max) / 2.0, (self.y_min + self.y_max) / 2.0

    @property
    def width(self) -> int | float:
        """Gets width of bounding box."""
        return self.x_max - self.x_min

    @property
    def height(self) -> int | float:
        """Gets height of bounding box."""
        return self.y_max - self.y_min

    @property
    def area(self) -> int | float:
        return self.width * self.height


@dataclass
class UIElement:
    """Represents a UI element on a web page."""

    attributes: Optional[str] = None
    value: Optional[str] = None
    type: Optional[str] = None

    bbox_pixels: Optional[BoundingBox] = None
    bbox: Optional[BoundingBox] = None

    is_clickable: bool = False
    is_editable: bool = None
    is_scrollable: bool = None
    is_visible: bool = False  # Assume visible if present in properties
    is_focused: bool = False

    bid: str = None  # BrowserGym ID
    uid: str = None

    def __post_init__(self):
        if self.bbox_pixels is not None and isinstance(self.bbox_pixels, dict):
            self.bbox_pixels = BoundingBox(**self.bbox_pixels)
        if self.bbox is not None and isinstance(self.bbox, dict):
            self.bbox = BoundingBox(**self.bbox)

    def __eq__(self, other):
        if not isinstance(other, UIElement):
            return NotImplemented
        return self.uid == other.uid

    def __hash__(self):
        return hash(self.uid)

    def __repr__(self):
        s = "UIElement("
        if self.bid:
            s += f"bid={self.bid}, "
        if self.type:
            s += f"""type="{self.type}", """
        if self.value:
            s += f"""value="{self.value}", """
        if self.attributes:
            s += f"""attributes="{self.attributes}", """
        if self.is_clickable:
            s += f"is_clickable={self.is_clickable}, "
        if self.is_editable is not None:
            s += f"is_editable={self.is_editable}, "
        if self.is_scrollable is not None:
            s += f"is_scrollable={self.is_scrollable}, "
        if self.is_visible:
            s += f"is_visible={self.is_visible}, "
        if self.is_focused:
            s += f"is_focused={self.is_focused}, "
        if self.bbox:
            s += f"bbox={self.bbox}, "
        if self.bbox_pixels:
            s += f"bbox_pixels={self.bbox_pixels}, "
        s = s.removesuffix(", ") + ")"
        return s


def _generate_ui_element_description(ui_element: UIElement, index: int = None) -> str:
    """Generate a description for a given UI element with important information.

    Args:
      ui_element: UI elements for the current screen.
      index: The numeric index for the UI element.

    Returns:
      The description for the UI element.
    """
    element_description = "UI element: {"
    if index is not None:
        element_description = f'UI element {index}: {{"index": {index}, '
    if ui_element.bid:
        element_description += f'"bid": "{ui_element.bid}", '
    if ui_element.type:
        element_description += f'"type": "{ui_element.type}", '
    if ui_element.value:
        element_description += f'"value": "{ui_element.value}", '
    if ui_element.attributes:
        element_description += f'"attributes": "{ui_element.attributes}", '
    if ui_element.is_clickable:
        element_description += f'"is_clickable": {ui_element.is_clickable}, '
    if ui_element.is_editable is not None:
        element_description += f'"is_editable": {ui_element.is_editable}, '
    if ui_element.is_scrollable is not None:
        element_description += f'"is_scrollable": {ui_element.is_scrollable}, '
    if ui_element.is_visible:
        element_description += f'"is_visible": {ui_element.is_visible}, '
    if ui_element.is_focused:
        element_description += f'"is_focused": {ui_element.is_focused}, '
    element_description = element_description.removesuffix(", ") + "}"
    return element_description


IGNORED_AXTREE_ROLES = ["LineBreak"]

IGNORED_AXTREE_PROPERTIES = (
    "editable",
    "readonly",
    "level",
    "settable",
    "multiline",
    "invalid",
    "focusable",
)


def _get_coord_str(coord, decimals):
    if isinstance(coord, str):
        coord = list(map(float, ast.literal_eval(coord)))

    coord_format = f".{decimals}f"
    coord_str = ",".join([f"{c:{coord_format}}" for c in coord])
    return f"({coord_str})"


def _process_bid(
    bid,
    extra_properties: dict = None,
    with_visible: bool = False,
    with_clickable: bool = False,
    with_center_coords: bool = False,
    with_bounding_box_coords: bool = False,
    with_som: bool = False,
    filter_visible_only: bool = False,
    filter_with_bid_only: bool = False,
    filter_som_only: bool = False,
    coord_decimals: int = 0,
):
    """
    Process extra attributes and attribute-based filters, for the element with the given bid.

    Returns:
        A flag indicating if the element should be skipped or not (due to filters).
        Attributes to be printed, as a list of "x=y" strings.
    """

    if extra_properties is None:
        if any(
            (
                with_visible,
                with_clickable,
                with_center_coords,
                with_bounding_box_coords,
                with_som,
                filter_visible_only,
                filter_with_bid_only,
                filter_som_only,
            )
        ):
            raise ValueError("extra_properties argument required")
        else:
            extra_properties = {}

    skip_element = False
    attributes_to_print = []

    if bid is None:
        if filter_with_bid_only:
            skip_element = True
        if filter_som_only:
            skip_element = True
        if filter_visible_only:
            pass

    # parse extra browsergym properties, if node has a bid
    else:
        if bid in extra_properties:
            node_vis = extra_properties[bid]["visibility"]
            node_bbox = extra_properties[bid]["bbox"]
            node_is_clickable = extra_properties[bid]["clickable"]
            node_in_som = extra_properties[bid]["set_of_marks"]
            node_is_visible = node_vis >= 0.5
            # skip non-visible nodes (if requested)
            if filter_visible_only and not node_is_visible:
                skip_element = True
            if filter_som_only and not node_in_som:
                skip_element = True
            # print extra attributes if requested (with new names)
            if with_som and node_in_som:
                attributes_to_print.insert(0, "som")
            if with_visible and node_is_visible:
                attributes_to_print.insert(0, "visible")
            if with_clickable and node_is_clickable:
                attributes_to_print.insert(0, "clickable")
            if with_center_coords and node_bbox is not None:
                x, y, __width, __height = node_bbox
                center = (x + __width / 2, y + __height / 2)
                attributes_to_print.insert(
                    0, f'center="{_get_coord_str(center, coord_decimals)}"'
                )
            if with_bounding_box_coords and node_bbox is not None:
                x, y, __width, __height = node_bbox
                box = (x, y, x + __width, y + __height)
                attributes_to_print.insert(
                    0, f'box="{_get_coord_str(box, coord_decimals)}"'
                )

    return skip_element, attributes_to_print


def _obs_to_ui_elements(obs: Dict[str, Any]) -> List[UIElement]:
    """
    Converts a BrowserGym observation's extra properties into a list of UIElement objects.
    """
    __ui_elements = []

    screen_height, screen_width = obs["screenshot"].shape[:2]  # (height, width)

    node_id_to_idx = {}
    for idx, node in enumerate(obs["axtree_object"]["nodes"]):
        node_id_to_idx[node["nodeId"]] = idx

    def dfs(
        node_idx: int, depth: int, parent_node_filtered: bool, parent_node_name: str
    ) -> str:
        tree_str = ""
        node = obs["axtree_object"]["nodes"][node_idx]
        skip_node = False
        filter_node = False
        node_role = node["role"]["value"]
        node_name = ""
        has_editable = False
        editable = False
        has_scrollable = False
        scrollable = False

        if node_role in IGNORED_AXTREE_ROLES:
            skip_node = True
            pass
        elif "name" not in node:
            skip_node = True
            pass
        else:
            node_name = node["name"]["value"]
            if "value" in node and "value" in node["value"]:
                node_value = node["value"]["value"]
            else:
                node_value = None
            bid = node.get("browsergym_id", None)
            attributes = []
            for _property in node.get("properties", []):
                if not "value" in _property:
                    continue
                if not "value" in _property["value"]:
                    continue

                prop_name = _property["name"]
                prop_value = _property["value"]["value"]

                if prop_name == "editable":
                    has_editable = True
                    if prop_value:
                        editable = True
                elif prop_name == "scrollable":
                    has_scrollable = True
                    if prop_value:
                        scrollable = True

                if prop_name in IGNORED_AXTREE_PROPERTIES:
                    continue
                elif prop_name in ("required", "focused", "atomic"):
                    if prop_value:
                        attributes.append(prop_name)
                else:
                    attributes.append(f"{prop_name}={repr(prop_value)}")

            if node_role == "generic" and not attributes:
                skip_node = True

            if node_role == "StaticText":
                if parent_node_filtered:
                    skip_node = True
                elif node_name in parent_node_name:
                    skip_node = True
            else:
                filter_node, extra_attributes_to_print = _process_bid(
                    bid,
                    extra_properties=obs["extra_element_properties"],
                    with_visible=False,
                    with_clickable=False,
                    with_center_coords=False,
                    with_bounding_box_coords=False,
                    with_som=False,
                    filter_visible_only=False,
                    filter_with_bid_only=False,
                    filter_som_only=False,
                    coord_decimals=0,
                )
                skip_node = skip_node or filter_node
                attributes = extra_attributes_to_print + attributes

            if bid is None:
                skip_node = True
            if not skip_node:
                ele = UIElement(
                    bid=bid,
                    uid=f"{bid}_{obs.get('url', 'None')}",
                )
                if node_role == "generic" and not node_name:
                    node_str = f"{node_role}"
                    ele.type = node_str
                else:
                    node_str = f"{node_role} {repr(node_name.strip())}"
                    ele.type = node_str.removesuffix(""" ''""")
                if node_value is not None:
                    ele.value = repr(node["value"]["value"])
                if attributes:
                    ele.attributes = ", ".join(attributes)
                extra_properties = obs["extra_element_properties"].get(bid, {})

                bbox_dict = extra_properties.get("bbox", None)
                if bbox_dict:
                    x, y, w, h = bbox_dict
                    bbox_pixels = BoundingBox(
                        x_min=max(int(x), 0),
                        y_min=max(int(y), 0),
                        x_max=min(int(x + w), screen_width),
                        y_max=min(int(y + h), screen_height),
                    )
                    bbox = BoundingBox(
                        x_min=max(x / screen_width, 0.0),
                        y_min=max(y / screen_height, 0.0),
                        x_max=min((x + w) / screen_width, 1.0),
                        y_max=min((y + h) / screen_height, 1.0),
                    )
                    ele.bbox = bbox
                    ele.bbox_pixels = bbox_pixels

                ele.is_clickable = extra_properties.get("clickable", False)
                ele.is_visible = extra_properties.get(
                    "visibility", 1.0
                ) > 0.5 or extra_properties.get("set_of_marks", False)
                ele.is_focused = bid == obs["focused_element_bid"]

                if has_editable:
                    ele.is_editable = editable
                if has_scrollable:
                    ele.is_scrollable = scrollable

                __ui_elements.append(ele)

        for child_node_id in node["childIds"]:
            if child_node_id not in node_id_to_idx or child_node_id == node["nodeId"]:
                continue
            child_depth = depth if skip_node else (depth + 1)
            child_str = dfs(
                node_id_to_idx[child_node_id],
                child_depth,
                parent_node_filtered=filter_node,
                parent_node_name=node_name,
            )
        return tree_str

    tree_str = dfs(0, 0, False, "")

    return __ui_elements


class WebDevice:
    """
    A class to interact with a WebArena environment, mimicking the Android Device class.
    Each website/domain can be treated as an "app".
    """

    def __init__(self, home_url: str):
        """
        Initializes the WebDevice by creating a BrowserGym environment.
        Args:
            task_id: The ID of the WebArena task (e.g., "webarena.shopping-1").
        """
        self.home_url = home_url
        self.env: gym.Env = None

        self.action_set = HighLevelActionSet(
            subsets=("bid", "coord", "nav", "tab", "chat")
        )

        self.connect()

    def connect(self):
        """Creates and resets the BrowserGym environment."""
        if self.env:
            self.env.close()

        self.env = gym.make(
            "browsergym/webarena.1",
            max_episode_steps=-1,
            action_mapping=self.action_set.to_python_code,
            headless=True,
            slow_mo=200,
        )
        self.reset()
        logger.info(
            "Successfully connected and reset environment. Start URL: %s",
            self.get_current_url(),
        )

    def disconnect(self):
        """Closes the BrowserGym environment."""
        if self.env:
            logger.info("Disconnecting from WebDevice.")
            self.env.close()
            self.env = None

    def reset(self):
        """Resets the environment to its initial state."""
        if not self.env:
            raise ConnectionError("Environment is not connected. Call connect() first.")
        obs, info = self.env.reset()
        self.goto(self.home_url)  # Navigate to the home URL

    def _step(self, action_str: str) -> None:
        """Internal method to perform a step in the environment."""
        if not self.env:
            raise ConnectionError("Environment is not connected.")
        obs, reward, terminated, truncated, info = self.env.step(action_str)
        return obs, reward, terminated, truncated, info

    def get_screenshot(self) -> Image.Image:
        """Returns the current page screenshot as a PIL Image."""
        obs, _, _, _, _ = self._step("noop(1)")
        return Image.fromarray(obs["screenshot"])

    def get_screen_size(self) -> Tuple[int, int]:
        """Returns the screen (width, height)."""
        screenshot = self.get_screenshot()
        return screenshot.size

    def get_current_url(self) -> str:
        """Returns the current URL of the active page."""
        obs, _, _, _, _ = self._step("noop(1)")
        return obs.get("url", "")

    def get_ui_elements(self) -> List[UIElement]:
        """Returns a list of UIElement objects from the current observation."""
        obs, _, _, _, _ = self._step("noop(1)")
        return _obs_to_ui_elements(obs)

    # --- Action Methods ---

    def click(self, element_bid: str):
        """Clicks on a UI element identified by its BrowserGym ID."""
        action_str = f'click("{element_bid}")'
        self._step(action_str)

    def type(self, element_bid: str, text: str, press_enter_after: bool = True):
        """
        Types text into a UI element identified by its BrowserGym ID.
        If press_enter_after is True, it simulates pressing the Enter key after typing.
        """
        action_str = f'type("{element_bid}", "{text}")'
        self._step(action_str)
        if press_enter_after:
            self.press("Enter")

    def hover(self, element_bid: str):
        """Hovers over a UI element identified by its BrowserGym ID."""
        action_str = f'hover("{element_bid}")'
        self._step(action_str)

    def press(self, key_combination: str):
        """
        Press a combination of keys. Accepts the logical key names that are emitted in the keyboardEvent.key property of the keyboard events: Backquote, Minus, Equal, Backslash, Backspace, Tab, Delete, Escape, ArrowDown, End, Enter, Home, Insert, PageDown, PageUp, ArrowRight, ArrowUp, F1 - F12, Digit0 - Digit9, KeyA - KeyZ, etc. You can alternatively specify a single character you'd like to produce such as "a" or "#". Following modification shortcuts are also supported: Shift, Control, Alt, Meta, ShiftLeft, ControlOrMeta. ControlOrMeta resolves to Control on Windows and Linux and to Meta on macOS.

        Examples:

        keyboard_press('Backspace')
        keyboard_press('ControlOrMeta+a')
        keyboard_press('Meta+Shift+t')
        page.keyboard.press("PageDown")
        """
        action_str = f'keyboard_press("{key_combination}")'
        self._step(action_str)

    def scroll(self, direction: str = "down"):
        """
        Scroll the page up or down.
        Args:
            direction: "down" or "up". Defaults to "down".
        """
        if direction not in ("down", "up"):
            raise ValueError("Direction must be 'down' or 'up'.")

        action_str = None
        if direction == "down":
            action_str = "scroll(0, 200)"  # Scroll down by 200 pixels
        else:
            action_str = "scroll(0, -200)"  # Scroll up by 200 pixels
        self._step(action_str)

    def new_tab(self):
        """Opens a new, empty browser tab."""
        action_str = "new_tab()"
        self._step(action_str)

    def tab_focus(self, tab_index: int):
        """
        Switches the browser's focus to a specific tab using its index.
        Args:
            tab_index: The index of the tab to focus on (0-based).
        """
        action_str = f"tab_focus({tab_index})"
        self._step(action_str)

    def close_tab(self):
        """Closes the currently active tab."""
        action_str = "tab_close()"
        self._step(action_str)

    def goto(self, url: str):
        """
        Navigates to a specific URL. Will wait for the page to load.
        Args:
            url: The URL to navigate to.
        """
        action_str = f'goto("{url}")'
        self._step(action_str)

    def go_back(self):
        """Navigates to the previously viewed page."""
        action_str = "go_back()"
        self._step(action_str)

    def go_forward(self):
        """Navigates to the next page (if a previous 'go_back' action was performed)."""
        action_str = "go_forward()"
        self._step(action_str)

    def stop(self, answer: str):
        """
        Issue this action when you believe the task is complete. If the objective is to find a text-based answer, provide the answer in the bracket. If you believe the task is impossible to complete, provide the answer as "N/A" in the bracket.
        """
        if answer != "N/A":
            self.send_msg_to_user(answer)
        else:
            self.report_infeasible("N/A")

    def send_msg_to_user(self, message: str):
        """
        Sends a message to the user.

        Examples:
        send_msg_to_user("Based on the results of my search, the city was built in 1751.")
        """
        action_str = f'send_msg_to_user("{message}")'
        self._step(action_str)

    def report_infeasible(self, reason: str):
        """
        Notifies the user that their instructions are infeasible.

        Examples:
        report_infeasible("I cannot follow these instructions because there is no email field in this form.")
        """
        action_str = f'report_infeasible("{reason}")'
        self._step(action_str)

    def get_open_pages_urls(self) -> List[str]:
        """
        Returns a list of URLs of all open pages in the browser.
        This is useful for debugging and understanding the current state of the browser.
        """
        obs, _, _, _, _ = self._step("noop(1)")
        return obs.get("open_pages_urls", [])

    def get_open_pages_titles(self) -> Tuple[str]:
        """
        Returns a list of titles of all open pages in the browser.
        This is useful for debugging and understanding the current state of the browser.
        """
        obs, _, _, _, _ = self._step("noop(1)")
        return obs.get("open_pages_titles", ())

    def get_active_page_index(self) -> int:
        """
        Returns the index of the currently active page in the browser.
        This is useful for debugging and understanding which page is currently focused.
        """
        obs, _, _, _, _ = self._step("noop(1)")
        active_page_indexs = obs.get("active_page_index", [])
        if len(active_page_indexs) > 0:
            return active_page_indexs[-1]
        return -1  # Return -1 if no active page or index is not available

    def get_current_page_title(self) -> str:
        """
        Returns the title of the currently active page in the browser.
        This is useful for debugging and understanding the current state of the browser.
        """
        obs, _, _, _, _ = self._step("noop(1)")
        active_page_indexs = obs.get("active_page_index", [])
        if len(active_page_indexs) > 0:
            active_page_index = active_page_indexs[-1]
            open_pages_titles = obs.get("open_pages_titles", ())
            if 0 <= active_page_index < len(open_pages_titles):
                return open_pages_titles[active_page_index]
        return "None"  # Return "None" if no active page or index is out of bounds


def is_element_available(ele: UIElement) -> bool:
    """
    Check if the UI element is available for interaction.
    """
    return ele.is_clickable or ele.is_scrollable or ele.is_editable


import numpy as np
import cv2


def add_ui_element_mark(
    screenshot: np.ndarray,
    ui_element: UIElement,
    index: int | str,
):
    """Add mark (a bounding box plus index) for a UI element in the screenshot.

    Args:
      screenshot: The screenshot as a numpy ndarray.
      ui_element: The UI element to be marked.
      index: The index for the UI element.
    """
    if ui_element.bbox_pixels:
        upper_left = ui_element.bbox_pixels.x_min, ui_element.bbox_pixels.y_min
        lower_right = ui_element.bbox_pixels.x_max, ui_element.bbox_pixels.y_max
        cv2.rectangle(
            screenshot,
            upper_left,
            lower_right,
            color=(0, 255, 0),
            thickness=3,
        )
        screenshot[
            upper_left[1] + 1 : upper_left[1] + 25,
            upper_left[0] + 1 : upper_left[0] + 35,
            :,
        ] = (255, 255, 255)
        cv2.putText(
            screenshot,
            str(index),
            (
                upper_left[0] + 1,
                upper_left[1] + 20,
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
    _height, _width, _ = screenshot.shape
    screenshot[_height - 30 : _height, _width - 150 : _width, :] = (255, 255, 255)
    cv2.putText(
        screenshot,
        label,
        (_width - 135, _height - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 0),
        thickness=3,
    )


if __name__ == "__main__":
    # Example usage and simple test
    logger.info("--- Testing WebDevice ---")

    # Ensure you have the WebArena environment set up correctly
    try:
        device = WebDevice(home_url="http://127.0.0.1:12001/")

        print(f"Initial URL: {device.get_current_url()}")

        print(f"Current page title: {device.get_current_page_title()}")

        # Get and show screenshot size
        width, height = device.get_screen_size()
        print(f"Screen size: {width}x{height}")

        # Stabilize and get UI elements
        ui_elements = device.get_ui_elements()
        print(f"Found {len(ui_elements)} UI elements after stabilization.")
        if ui_elements:
            print("First 5 UI elements:")
            for i, el in enumerate(ui_elements[:5]):
                print(el)

        print("\nPerforming a click")
        device.click(ui_elements[0].bid)  # Click the first UI element

        # Get new state
        print(f"New URL: {device.get_current_url()}")
        print(f"Current page title: {device.get_current_page_title()}")
        ui_elements_after_click = device.get_ui_elements()
        print(f"Found {len(ui_elements_after_click)} UI elements after click.")

        # Test navigation
        print("\nTesting navigation...")
        device.go_back()
        print(f"URL after back(): {device.get_current_url()}")
        print(f"Current page title: {device.get_current_page_title()}")

        device.disconnect()
        print("\n--- Test complete ---")

    except Exception as e:
        print("\n--- Test failed ---")
        print(f"An error occurred: {e}")
        print(
            "Please ensure you have browsergym and its webarena dependencies installed and configured."
        )
