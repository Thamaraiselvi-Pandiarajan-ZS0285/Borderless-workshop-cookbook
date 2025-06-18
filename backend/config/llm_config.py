from backend.config.dev_config import *


config_list = [
                {
            "model": AZURE_OPENAI_DEPLOYMENT_NAME,
            "api_type": "azure",
            "api_key": AZURE_OPENAI_API_KEY,
            "base_url": AZURE_OPENAI_ENDPOINT,
            "api_version": AZURE_OPENAI_API_VERSION,
                 }
             ]
LLM_CONFIG = {
    "config_list": config_list,
    "temperature": 0.1,
    "timeout": 120,
}

