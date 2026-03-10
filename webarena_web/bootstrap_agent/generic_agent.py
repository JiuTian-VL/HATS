"""
GenericAgent implementation for AgentLab

This module defines a `GenericAgent` class and its associated arguments for use in the AgentLab framework. \
The `GenericAgent` class is designed to interact with a chat-based model to determine actions based on \
observations. It includes methods for preprocessing observations, generating actions, and managing internal \
state such as plans, memories, and thoughts. The `GenericAgentArgs` class provides configuration options for \
the agent, including model arguments and flags for various behaviors.
"""

from copy import deepcopy
from dataclasses import asdict, dataclass
from warnings import warn

from bgym import Benchmark
from bgym import HighLevelActionSetArgs
from browsergym.experiments.agent import Agent, AgentInfo

from agentlab.agents import dynamic_prompting as dp
from agentlab.agents.agent_args import AgentArgs
from agentlab.llm.chat_api import BaseModelArgs
from agentlab.llm.llm_utils import Discussion, ParseError, SystemMessage, retry
from agentlab.llm.tracking import cost_tracker_decorator

from .generic_agent_prompt import GenericPromptFlags, MainPrompt

from agentlab.llm.chat_api import OpenAIModelArgs
import os
import time
from webarena_web.webarena_device import WebDevice,_obs_to_ui_elements
import re

def str2nums(s: str) -> list[int]:
  """
  使用正则表达式从字符串中提取所有整数（包括负数）。

  Args:
    s: 输入的字符串。

  Returns:
    一个包含所有从字符串中提取的整数的列表。
  """
  num_strings = re.findall(r'-?\d+', s)
  return [int(num) for num in num_strings]

class GenericAgent(Agent):

    def __init__(
        self,
        home_url: str,
        max_retry: int = 4,
    ):
        chat_model_args = OpenAIModelArgs(
            model_name=os.getenv("OPENAI_API_MODEL", "gpt-4o"),
            max_total_tokens=128_000,
            max_input_tokens=128_000,
            max_new_tokens=16384,
            vision_support=True,
        )

        print(f"Using model: {chat_model_args.model_name}")

        self.chat_llm = chat_model_args.make_model()
        self.chat_model_args = chat_model_args
        self.max_retry = max_retry

        self.flags = GenericPromptFlags(
            obs=dp.ObsFlags(
                use_html=False,
                use_ax_tree=True,
                use_focused_element=True,
                use_error_logs=True,
                use_history=True,
                use_past_error_logs=False,
                use_action_history=True,
                use_think_history=True,
                use_diff=False,
                html_type="pruned_html",
                use_screenshot=False,
                use_som=False,
                extract_visible_tag=True,
                extract_clickable_tag=True,
                extract_coords="False",
                filter_visible_elements_only=False,
            ),
            action=dp.ActionFlags(
                action_set=HighLevelActionSetArgs(
                    subsets=["bid"],
                    multiaction=False,
                ),
                long_description=False,
                individual_examples=False,
            ),
            use_plan=False,
            use_criticise=False,
            use_thinking=True,
            use_memory=False,
            use_concrete_example=True,
            use_abstract_example=True,
            use_hints=True,
            enable_chat=False,
            max_prompt_tokens=40_000,
            be_cautious=True,
            extra_instructions=None,
        )
        self.action_set = self.flags.action.action_set.make_action_set()
        self._obs_preprocessor = dp.make_obs_preprocessor(self.flags.obs)

        self._check_flag_constancy()
        self.reset(seed=42)
        self.step_interval = 3
        self.home_url = home_url
        self.device = WebDevice(home_url=home_url)

    def run(
        self,
        task_goal: str,
        max_rounds: int = 30,
        step_interval: float = 3.0,
    ) -> list[dict[str,]]:
        self.reset()
        print(f"Running task: {task_goal}")
        self.step_interval = step_interval
        step_datas = []
        for i in range(max_rounds):
            print(f"\nRound {i + 1}/{max_rounds}")
            stop, _step_data = self.step(task_goal)
            print(f"Reasoning: {_step_data['reasoning']}")
            print(f"Action: {_step_data['converted_action']}")
            step_datas.append(_step_data)
            if stop:
                break
            time.sleep(self.step_interval)
        print("Task completed or maximum rounds reached.")
        return step_datas

    def step(self, goal: str) -> tuple[bool, dict[str,]]:
        step_data = {
            "screenshot": None,
            "reasoning": "None",
            "converted_action": "None",
            "target_element": None,
            "current_page_title": "None",
            "current_url": "None",
            "agent_info": None,
        }
        obs, reward, terminated, truncated, info = self.device.env.step("noop(1)")
        obs=self._obs_preprocessor(obs)
        step_data["screenshot"]=obs["screenshot"] # np.array
        step_data["current_url"] = obs.get("url", "None")
        active_page_indexs = obs.get("active_page_index", [])
        if len(active_page_indexs) > 0:
            active_page_index = active_page_indexs[-1]
            open_pages_titles = obs.get("open_pages_titles", ())
            if 0 <= active_page_index < len(open_pages_titles):
                step_data["current_page_title"] = open_pages_titles[active_page_index]
        if 'goal_object' in obs and obs['goal_object'] is not None:
            obs['goal_object']=({'type': 'text', 'text': goal},)
        if 'goal' in obs and obs['goal'] is not None:
            obs['goal'] = goal
        if 'chat_messages' in obs and obs['chat_messages'] is not None:
            obs['chat_messages'][1]['message']=goal
        action, agent_info = self.get_action(obs)
        if "click" in action or "fill" in action or "hover" in action or "select_option" in action:
            _ui_elements=_obs_to_ui_elements(obs)
            nums=str2nums(action)
            if len(nums) >0:
                bid=nums[0]
                for ele in _ui_elements:
                    if str(ele.bid) == str(bid):
                        step_data["target_element"] = asdict(ele)
                        break
                else:
                    print(f"Warning: No UI element found for bid {bid} in action '{action}'")
            else:
                print(f"Warning: No numbers found in action '{action}'")
        step_data["agent_info"] = asdict(agent_info)
        step_data["converted_action"] = action
        step_data["reasoning"] = self.thoughts[-1] if self.thoughts else "None"
        if action is None or "noop()" in action:
            return True, step_data
        _,_,_,_,_=self.device.env.step(action)
        return False, step_data

    def obs_preprocessor(self, obs: dict) -> dict:
        return self._obs_preprocessor(obs)

    # @cost_tracker_decorator
    def get_action(self, obs):

        self.obs_history.append(obs)
        main_prompt = MainPrompt(
            action_set=self.action_set,
            obs_history=self.obs_history,
            actions=self.actions,
            memories=self.memories,
            thoughts=self.thoughts,
            previous_plan=self.plan,
            step=self.plan_step,
            flags=self.flags,
        )

        max_prompt_tokens, max_trunc_itr = self._get_maxes()

        system_prompt = SystemMessage(dp.SystemPrompt().prompt)

        human_prompt = dp.fit_tokens(
            shrinkable=main_prompt,
            max_prompt_tokens=max_prompt_tokens,
            model_name=self.chat_model_args.model_name,
            max_iterations=max_trunc_itr,
            additional_prompts=system_prompt,
        )
        try:
            chat_messages = Discussion([system_prompt, human_prompt])
            ans_dict = retry(
                self.chat_llm,
                chat_messages,
                n_retry=self.max_retry,
                parser=main_prompt._parse_answer,
            )
            ans_dict["busted_retry"] = 0
            ans_dict["n_retry"] = (len(chat_messages) - 3) / 2
        except ParseError as e:
            ans_dict = {
                "action": None,
                "n_retry": self.max_retry + 1,
                "busted_retry": 1,
            }

        stats = self.chat_llm.get_stats()
        stats["n_retry"] = ans_dict["n_retry"]
        stats["busted_retry"] = ans_dict["busted_retry"]

        self.plan = ans_dict.get("plan", self.plan)
        self.plan_step = ans_dict.get("step", self.plan_step)
        self.actions.append(ans_dict["action"])
        self.memories.append(ans_dict.get("memory", None))
        self.thoughts.append(ans_dict.get("think", None))

        agent_info = AgentInfo(
            think=ans_dict.get("think", None),
            chat_messages=chat_messages,
            stats=stats,
            extra_info={
                "chat_model_args": asdict(self.chat_model_args),
                "obs": obs,
                "ans_dict": ans_dict,
            },
        )
        return ans_dict["action"], agent_info

    def reset(self, seed=None):
        self.seed = seed
        self.plan = "No plan yet"
        self.plan_step = -1
        self.memories = []
        self.thoughts = []
        self.actions = []
        self.obs_history = []

    def _check_flag_constancy(self):
        flags = self.flags
        if flags.obs.use_som:
            if not flags.obs.use_screenshot:
                warn(
                    """
Warning: use_som=True requires use_screenshot=True. Disabling use_som."""
                )
                flags.obs.use_som = False
        if flags.obs.use_screenshot:
            if not self.chat_model_args.vision_support:
                warn(
                    """
Warning: use_screenshot is set to True, but the chat model \
does not support vision. Disabling use_screenshot."""
                )
                flags.obs.use_screenshot = False
        return flags

    def _get_maxes(self):
        maxes = (
            self.flags.max_prompt_tokens,
            self.chat_model_args.max_total_tokens,
            self.chat_model_args.max_input_tokens,
        )
        maxes = [m for m in maxes if m is not None]
        max_prompt_tokens = min(maxes) if maxes else None
        max_trunc_itr = (
            self.flags.max_trunc_itr
            if self.flags.max_trunc_itr
            else 20  # dangerous to change the default value here?
        )
        return max_prompt_tokens, max_trunc_itr

if __name__ == "__main__":
    print("This module is not meant to be run directly.")
    print("Please use it as part of the AgentLab framework.")
    
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

    os.environ["OPENAI_API_KEY"] = "sk-123"
    os.environ["OPENAI_BASE_URL"] = "https://api.openai.com/v1"
    os.environ["OPENAI_API_MODEL"] = "gpt-4o"
    
    task="Randomly click on a button on the page. Then tell me the title of the page."
    home_url = "https://www.baidu.com"
    agent = GenericAgent(home_url=home_url)
    execution_trajectory_data=agent.run(task, max_rounds=3, step_interval=2.0)
    for i,step_data in enumerate(execution_trajectory_data):
        print(f"\nStep {i + 1}:")
        print(f"screenshot shape: {step_data['screenshot'].shape if step_data['screenshot'] is not None else 'screenshot is None'}")
        print(f"reasoning: {step_data['reasoning']}")
        print(f"converted_action: {step_data['converted_action']}")
        print(f"target_element: {step_data['target_element']}")
        print(f"current_page_title: {step_data['current_page_title']}")
        print(f"current_url: {step_data['current_url']}")

#python -m webarena_web.agent.generic_agent