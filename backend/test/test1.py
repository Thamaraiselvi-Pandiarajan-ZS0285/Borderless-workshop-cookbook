from autogen import ConversableAgent
from backend.config.llm_config import LlmConfig

test_agent = ConversableAgent(name="TestAgent", llm_config=LlmConfig().llm_config)

def test_llm():
    response = test_agent.generate_reply(messages=[{"role": "user", "content": "Hello"}])
    print("Test LLM response:", response)
    print("LLM config:", test_agent.llm_config)

test_llm()
