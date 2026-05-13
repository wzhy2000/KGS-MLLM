import os

# ========== Model and API configuration ==========
DEFAULT_MODEL = 'gemini-3-pro-preview'                                              #gpt-5.2-chat-latest, claude-opus-4-5-20251101-thinking, grok-4-0709
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 0.0
DEFAULT_MAX_TOKENS = 4096

API_KEY = os.environ.get("API_KEY", "your_default_api_key_here")                    #your_default_api_key
PROXY_API_URL = os.environ.get("PROXY_API_URL", "your_default_api_url_here")        #your_default_api_url

# ========== Base Path Configuration ==========
# Please modify to your actual workspace root directory
BASE_WORKSPACE = r"your_actual_workspace_root_directory"                            #your actual workspace root directory