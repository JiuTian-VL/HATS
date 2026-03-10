# INPUT: index
GENERATE_INPUT_CONTENT_PROMPT = """Generate appropriate input content for the input field marked with a green bounding box and labeled as number {index} in the screenshot. Your generated content should be suitable for the usage scenario shown in this screenshot and represent what real users would typically enter in daily use. Your response should contain only the text that needs to be entered into the input field."""


# INPUT: trajectory_information
TRAJECTORY_TO_INSTRUCTION_PROMPT = """**You are an expert specializing in inferring specific user tasks based on changes observed in mobile phone screenshots within an interaction trajectory.** I will provide you with an interaction trajectory containing the following information:

1.  **Action per Step:** The action performed at each step, chosen from one of these types: `click`, `double_click`, `long_click`, `scroll_up`, `scroll_down`, `scroll_left`, `scroll_right`, `input`. If the action is `input`, the input text will also be provided. Associated with each action (except scrolls) is information about the targeted UI element, including attributes like `content_description`, `text`, `resource_id`, etc.
2.  **Screenshots:** A screenshot taken *before* each action (labeled "Step i" in the bottom-right corner, indicating it precedes the i-th action) and a final screenshot taken *after* the last action (labeled "Final" in the bottom-right). The pre-action screenshots will feature a green bounding box (marked with the step number 'i' in the top-left corner, also identifying the i-th UI element.) highlighting the element being interacted with. **Pay close attention to the content within or associated with the green bounding box and the changes between the 'before' and 'after' screenshots for each step.**
3.  **Package Name:** The `package_name` of the application active in the current screenshot (you can use this to infer the app's common name).

Note: In Sub-Instruction, Analysis, Knowledge, and High-Level-Instruction, do not use the numerical marker (from the top-left of the green box) to refer to the UI element. Instead, if you need to refer to the element, use a more natural description based on its visual characteristics or text content (e.g., "the 'Settings' icon", "the text input field labeled 'Username'").

**Your task involves two main parts:**

**Part 1: Analysis and Step Selection**
*   Analyze the *entire* provided trajectory (actions and screenshot changes).
*   Identify a **logically coherent subsequence of steps** within the trajectory that represents a complete and reasonable user task.
*   You must select actions that follow a clear and rational sequence toward achieving a specific goal.
*   **Eliminate** any steps from the original trajectory that are redundant, irrelevant, irrational, or unnecessary for completing the identified task.

**Part 2: Output Generation**
Based on your analysis and step selection, devise the specific user goal or task. Your output **must** include five parts:

*   **Sub-Instruction (List[str]):** For **each step** in the *original* trajectory, generate a natural language instruction corresponding to the action performed, based on the UI changes observed. This instruction should be concise, clear, actionable, and *must* incorporate key specific details visible in the screenshot, such as filenames, times, text content, or other relevant identifiers associated with the interacted element. Examples: "Scroll left to open the app drawer, revealing all installed applications.", "Tap on the chat interface labeled 'General Discussion'.", "Input the username 'Agent' into the username field."
*   **Analysis (List[str]):** For **each step** in the *original* trajectory, provide an analysis of the potential subsequent actions or user intent, based on the UI changes resulting from the action and its Sub-Instruction. This analysis should involve step-by-step reasoning, considering the observed screen state and what actions become possible or logical next. Example: "After tapping the '+' button, a menu with options like 'New Document' and 'New Folder' appeared. I can create something new. Next step would be to tap 'New Document', which might then prompt for a filename."
*   **Knowledge (List[str]):** For **each step `i`** in the *original* trajectory, describe the general *functionality* of the UI element interacted with in that step. This description should be inferred by comparing the screenshot *before* action `i` (labeled "Step i") and the screenshot *after* action `i` (which will be the screenshot labeled "Step i+1", or "Final" for the last action). The description should be concise (1-2 sentences), focus on the general function revealed by the interaction (e.g., "Opens a settings menu," "Navigates back," "Selects an item," "Confirms an action"), avoid specific details unless necessary for clarity, and use generic terms like "this element" or "this button" or a functional description (e.g., "the search icon"). The length of this list must be equal to the number of steps in the *original* trajectory. Example: "Tapping this element initiates a search and displays matching results."
*   **Selected-Step-ID (List[int]):** A list containing the integer step IDs (the 'i' from 'Step i') of the actions you **selected** in Part 1 as part of the coherent, logical task sequence. The IDs must be listed in the chronological order they appear in the selected subsequence.
*   **High-Level-Instruction (str):** Based **only** on the **selected subsequence of steps** identified in Part 1, formulate a single, high-level instruction describing the overall task the user was likely trying to achieve *through those selected steps*. This instruction should correspond directly and efficiently to the selected sequence. It can be either:
    *   **Task-Oriented:** Describe the sequence of selected actions to achieve a specific goal.
    *   **Question-Oriented:** Describe the sequence of selected actions to find the answer to a specific question.
    Examples:
    1.  "Mark the 'Fruit salad' recipe as a favorite in the Broccoli app." (Assuming steps related to finding and favoriting the recipe were selected).
    2.  "In the 'Fruit salad' recipe within the Broccoli app, what fruits are listed?" (Assuming steps related to finding and viewing the recipe details were selected).
    Ensure the High-Level-Instruction is actionable, includes all crucial specific details (like filenames, relevant times, required text, etc.) relevant to the *selected* steps, and explicitly mentions the inferred **app name** (e.g., "Broccoli app," not `com.flauschcode.broccoli`).

You must return **only** a JSON dictionary in the following format:

```json
{{
  "Sub-Instruction": List[str],
  "Analysis": List[str],
  "Knowledge": List[str],
  "Selected-Step-ID": List[int],
  "High-Level-Instruction": str
}}
```

The trajectory information will be provided below:
`{trajectory_information}`

**RETURN ME THE DICTIONARY I ASKED FOR.**
"""


# INPUTS: task_goal, history, ui_elements, knowledge
REASONING = """## Role Definition
You are an Android operation AI that fulfills user requests through precise screen interactions.
The current screenshot and the same screenshot with bounding boxes and labels added are also given to you.

## Action Catalog
Available actions (STRICT JSON FORMAT REQUIRED):
1. Status Operations:
   - Task Complete: {{"action_type": "status", "goal_status": "complete"}}
   - Task Infeasible: {{"action_type": "status", "goal_status": "infeasible"}}
2. Information Actions:
   - Answer Question: {{"action_type": "answer", "text": "<answer_text>"}}
3. Screen Interactions:
   - Tap Element: {{"action_type": "click", "index": <visible_index>}}
   - Long Press: {{"action_type": "long_press", "index": <visible_index>}}
   - Scroll: Scroll the screen or a specific scrollable UI element. Use the `index` of the target element if scrolling a specific element, or omit `index` to scroll the whole screen. {{"action_type": "scroll", "direction": <"up"|"down"|"left"|"right">, "index": <optional_target_index>}}
4. Input Operations:
   - Text Entry: {{"action_type": "input_text", "text": "<content>", "index": <text_field_index>}}
   - Keyboard Enter: {{"action_type": "keyboard_enter"}}
5. Navigation:
   - Home Screen: {{"action_type": "navigate_home"}}
   - Back Navigation: {{"action_type": "navigate_back"}}
6. System Actions:
   - Wait Refresh: {{"action_type": "wait"}}

## Current Objective
User Goal: {task_goal}

## Execution Context
Action History:
{history}

Visible UI Elements (Only interact with *visible=true elements):
{ui_elements}

## Core Strategy
1. Path Optimization:
   - To open an application, prioritize navigating to the app drawer (often accessed by swiping up from the home screen or tapping an 'All Apps' icon), locating the application icon within the drawer, and tapping it.
   - Always use the `input_text` action for entering text into designated text fields.
   - Verify element visibility (`visible=true`) before attempting any interaction (click, long_press, input_text). Do not interact with elements marked `visible=false`.
   - Use `scroll` when necessary to bring off-screen elements into view. Prioritize scrolling specific containers (`index` provided) over full-screen scrolls if possible.

2. Error Handling Protocol:
   - Switch approach after ≥ 2 failed attempts
   - Prioritize scrolling (`scroll` action) over force-acting on invisible elements
   - If an element is not visible, use `scroll` in the likely direction (e.g., 'down' to find elements below the current view).
   - Try opposite scroll direction if initial fails (up/down, left/right)

3. Information Tasks:
   - MANDATORY: Use `answer` action for questions
   - Verify data freshness (e.g., check calendar date)

## Expert Techniques
Here are some tips for you:
{knowledge}

## Response Format
STRICTLY follow:
Reasoning: [Step-by-step analysis covering:
           - Visibility verification
           - History effectiveness evaluation
           - Alternative approach comparison
           - Consideration of scrolling if needed]
Action: [SINGLE JSON action from catalog]

Generate response:
"""

# INPUTS: task_goal, before_ui_elements, after_ui_elements, action, reasoning
SUMMARY = """
Goal: {task_goal}

Before screenshot elements:
{before_ui_elements}

After screenshot elements:
{after_ui_elements}

Action: {action}
Reasoning: {reasoning}

Provide a concise single-line summary (under 50 words) of this step by comparing screenshots and action outcome. Include:
- What was intended
- Whether it succeeded
- Key information for future actions
- Critical analysis if action/reasoning was flawed
- Important data to remember across apps

For actions like 'answer' or 'wait' with no screen change, assume they worked as intended.

Summary:
"""

# INPUT: high_level_instruction, exploration_trajectory_information, gui_agent_trajectory_information
COUNT_EXPLORATION_STEPS_MATCHED_PROMPT = """**Objective:**

You are tasked with evaluating the execution trace of a GUI Agent against a reference exploration trajectory. Your goal is to determine which specific steps from the reference **Exploration Trajectory** were successfully matched by one or more steps in the **GUI Agent Trajectory**. You need to provide the count of matched Exploration steps, the indices of these matched Exploration steps, and the indices of the corresponding GUI Agent steps that performed the match.

**Input Information:**

You will be provided with the following:

1.  **High-Level Instruction:** The overall task or goal assigned to the GUI Agent. This instruction was originally derived from the Exploration Trajectory.
    ```
    {high_level_instruction}
    ```

2.  **Exploration Trajectory:** A sequence of steps representing an ideal or human-demonstrated path to achieve the high-level instruction. Each step is numbered (e.g., "Step 1:", "Step 2:") and typically includes a low-level instruction (sub-goal) and the specific action taken on a UI element.
    ```
    {exploration_trajectory_information}
    ```

3.  **GUI Agent Trajectory:** The sequence of steps the GUI Agent actually executed while attempting to follow the High-Level Instruction. Each step includes the agent's reasoning and the action it performed.
    ```
    {gui_agent_trajectory_information}
    ```


**Task:**

1.  Carefully compare the **GUI Agent Trajectory** against the **Exploration Trajectory**.
2.  Identify each distinct step from the **Exploration Trajectory** that has been successfully matched by at least one step in the **GUI Agent Trajectory**.
3.  Record the step numbers (the `i` in "Step i:") of these matched **Exploration Trajectory** steps.
4.  Record the step numbers (the `i` in "Step i:") of the **GUI Agent Trajectory** steps that successfully performed a match corresponding to an Exploration Trajectory step.
5.  Count the total number of unique matched **Exploration Trajectory** steps.

**Matching Rule (Very Important):**

*   A step in the Exploration Trajectory is considered "matched" if one or more steps in the GUI Agent Trajectory successfully perform the equivalent action or achieve the same sub-goal described in that Exploration Trajectory step.
*   **Crucially:** If multiple steps in the GUI Agent Trajectory correspond to the *same single step* in the Exploration Trajectory (e.g., the agent tries scrolling twice to achieve one scroll action in the exploration, or fails and retries an action), these multiple agent steps only count as **one (1) match** towards that *single* Exploration Trajectory step (affecting `match_num` and `matched_exploration_id`).
*   However, the indices of *all* GUI Agent steps that successfully contribute to matching *any* Exploration step should be included in the `matched_gui_agent_id` list.

**Output Requirement:**

Provide your response **only** in the following JSON format. Do not include any explanations, reasoning, or other text outside the JSON structure.

```json
{{
  "match_num": <int>,
  "matched_exploration_id": [<int>, <int>, ...],
  "matched_gui_agent_id": [<int>, <int>, ...]
}}
```

*   `match_num`: An integer representing the total count of unique **Exploration Trajectory** steps that were matched.
*   `matched_exploration_id`: A list of integers, where each integer is the step number (`i` from "Step i:") of a matched step *from the **Exploration Trajectory***. The list should contain only the indices of the matched unique Exploration steps, sorted in ascending order.
*   `matched_gui_agent_id`: A list of integers, where each integer is the step number (`i` from "Step i:") of a step *from the **GUI Agent Trajectory*** that successfully matched *any* step in the Exploration Trajectory. This list should contain the indices of *all* such matching agent steps, sorted in ascending order.

**Constraints:**
*   The value of `match_num` must be equal to the length of the `matched_exploration_id` list (`match_num == len(matched_exploration_id)`).
*   The length of `matched_gui_agent_id` may be greater than or equal to `match_num`, as multiple agent steps can match a single exploration step, or different agent steps can match different exploration steps.
"""

# INPUT: high_level_instruction, exploration_trajectory_information, gui_agent_trajectory_information, matched_low_level_instructions, matched_gui_agent_steps
REFINE_HIGH_LEVEL_INSTRUCTION_PROMPT = """**You are an expert AI assistant specializing in refining high-level instructions for GUI Agents.** Your goal is to improve a given `high_level_instruction` so that when a GUI Agent executes it, the resulting trajectory more closely matches a reference `exploration_trajectory`.

**Input Information:**

You will be provided with the following context:

1.  **Initial High-Level Instruction:** The instruction previously given to the GUI Agent.
    ```
    {high_level_instruction}
    ```

2.  **Exploration Trajectory:** The ideal sequence of steps (low-level instructions and actions) to achieve the underlying user goal. This represents the "ground truth" path.
    ```
    {exploration_trajectory_information}
    ```

3.  **GUI Agent Trajectory:** The actual sequence of steps (reasoning and actions) taken by the GUI Agent when attempting to follow the *Initial High-Level Instruction*.
    ```
    {gui_agent_trajectory_information}
    ```

4.  **Matched Exploration Steps:** A list of the low-level instructions from the `Exploration Trajectory` that the GUI Agent *successfully* executed during its run.
    ```
    {matched_low_level_instructions}
    ```

5.  **Matched GUI Agent Steps:** A list of the reasoning/action steps from the `GUI Agent Trajectory` that corresponded to the matched exploration steps.
    ```
    {matched_gui_agent_steps}
    ```

**Your Task:**

Analyze the discrepancy between the `Exploration Trajectory` and the `GUI Agent Trajectory`. Identify the steps in the `Exploration Trajectory` that were *not* matched by the agent (i.e., steps present in `exploration_trajectory_information` but whose low-level instructions are absent from `matched_low_level_instructions`). Examine the `GUI Agent Trajectory` (especially its reasoning and actions around the points of deviation or failure) to understand *why* the agent failed to match these steps.

Based on this analysis, **rewrite and refine the `Initial High-Level Instruction`**. The goal of the refined instruction is to guide the agent more effectively towards executing the *entire* sequence described in the `Exploration Trajectory`.

**Refinement Guidelines:**

*   **Increase Specificity:** Add crucial details from the *unmatched* steps of the `Exploration Trajectory` into the high-level instruction. This might include specific UI element descriptions (text, content description), types of actions (e.g., `long_click` instead of just `click`, `scroll_up` instead of generic navigation), or required input text.
*   **Clarify Sequence:** Ensure the sequence of actions described in the refined instruction logically follows the `Exploration Trajectory`, especially where the agent got confused or performed actions out of order.
*   **Address Agent Errors:** If the agent's reasoning indicates a misunderstanding of the goal or the UI, adjust the instruction's wording to prevent that misinterpretation. For example, if the agent used `navigate_back` when a specific button (`Home`) should have been pressed, clarify this.
*   **Maintain Goal Integrity:** The refined instruction must still represent the overall goal achievable by completing the full `Exploration Trajectory`.
*   **Use App Name:** Explicitly mention the inferred application name (e.g., "Broccoli app") as derived from the `package_name` in the trajectories.
*   **Be Actionable:** The instruction should remain a command or description of a task for the agent to perform.

**Output Requirement:**

Provide **only** the refined high-level instruction as a single JSON string value. Do not include explanations, analysis, or any other text outside the JSON structure.

```json
{{
  "refined_high_level_instruction": "<Your refined high-level instruction here>"
}}
```
"""
