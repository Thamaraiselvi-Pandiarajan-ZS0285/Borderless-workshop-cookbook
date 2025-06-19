from backend.config.dev_config import *

class LlmConfig():
    def __init__(self):
        self.llm_config= {
                    "config_list": [{
                        "model": AZURE_OPENAI_DEPLOYMENT,
                        "api_type": AZURE_API_TYPE,
                        "api_key": AZURE_OPENAI_API_KEY,
                        "base_url": AZURE_OPENAI_ENDPOINT,
                        "api_version": AZURE_OPENAI_API_VERSION
                    }],
                    "temperature": TEMPERATURE,
                }

        self.embedding_config = {
            "config_list": [{
                "model": AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
                "api_type": AZURE_API_TYPE,
                "api_key": AZURE_OPENAI_API_KEY,
                "base_url": AZURE_OPENAI_ENDPOINT,
                "api_version": AZURE_OPENAI_API_VERSION
            }],
            "temperature": TEMPERATURE,
        }