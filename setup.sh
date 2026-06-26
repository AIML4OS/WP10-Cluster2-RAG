# IMPORTANTA: Run using ´source setup.sh´

# Setup uv and kernel
uv sync
source .venv/bin/activate
uv run python -m ipykernel install --user --name ragenv --display-name "rag"

# Set up model token
read -s -p "Enter LLM endpoint token: " LLM_TOKEN_INPUT
echo
export LLM_TOKEN="$LLM_TOKEN_INPUT"

