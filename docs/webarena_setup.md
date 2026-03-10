# Additional WebArena Setup

This project is built on AgentLab / WebArena. Please complete the base environment first by following the official documentation:

- https://github.com/ServiceNow/AgentLab

## 1. Install Dependencies

Installing in the HATS environment is recommended:

```bash
pip install agentlab
```

Then install WebArena-related dependencies and data according to the AgentLab documentation.

## 2. Modify Upstream Library Configuration

In your AgentLab/WebArena environment, it is recommended to apply the following changes:

1. `webarena.llms.providers.openai_utils`

- At the location where the OpenAI client is created, add:
  - `base_url=os.getenv("OPENAI_BASE_URL_WA", "https://api.openai.com/v1")`
- Replace `OPENAI_API_KEY` with `OPENAI_API_KEY_WA`

2. `webarena.evaluation_harness.helper_functions`

- Replace the hardcoded model (for example, `gpt-4-1106-preview`) with:
  - `os.getenv("OPENAI_API_MODEL_WA", "gpt-4o")`

3. `OpenAIChatModel` in `agentlab.llm.chat_api`

- During initialization, add `client_args` and include `base_url`

## 3. Environment Variables Before Running

Before running any web-related scripts, execute:

```bash
source webarena_web/exp_env.sh
```

Then run exploration or data conversion scripts.
