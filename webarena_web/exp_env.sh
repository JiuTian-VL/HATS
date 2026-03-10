# Usage: source exp_env.sh
# Or: . exp_env.sh

# after `pip install agentlab`, follow the instruction of https://github.com/ServiceNow/AgentLab to install webarena's things before use this script

BASE_URL="http://127.0.0.1"

# webarena environment variables (change ports as needed)
export WA_SHOPPING="$BASE_URL:12001/" # maybe WA_SHOPPING="$BASE_URL:12001"
export WA_SHOPPING_ADMIN="$BASE_URL:12002/admin"
export WA_REDDIT="$BASE_URL:12003" # maybe WA_REDDIT="$BASE_URL:12003/forums/all"
export WA_GITLAB="$BASE_URL:12004" # maybe WA_GITLAB="$BASE_URL:12004/explore"
export WA_WIKIPEDIA="$BASE_URL:12005/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export WA_MAP="$BASE_URL:12006"
export WA_HOMEPAGE="$BASE_URL:12007"

# if your webarena instance offers the FULL_RESET feature (optional)
export WA_FULL_RESET="$BASE_URL:12008"

# 用于webarena的自动评分，除了key和url可能需要修改，不建议修改模型
export OPENAI_API_KEY_WA="sk-123"
export OPENAI_BASE_URL_WA="https://api.openai.com/v1"
export OPENAI_API_MODEL_WA="gpt-4o"

# 我们要在webarena评测的模型，可以修改成ms-swift部署的模型
export OPENAI_API_KEY="sk-123"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OPENAI_API_MODEL="gpt-4o"

# export AGENTLAB_EXP_ROOT="/path/to/agentlab_results"  # defaults to $HOME/agentlab_results
