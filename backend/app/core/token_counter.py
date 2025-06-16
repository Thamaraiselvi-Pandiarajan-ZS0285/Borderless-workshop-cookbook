

class TokenCounter:
    def __init__(self, model_name:str, system_prompt:str, user_prompt:str, visual_token:bytes|None):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.visual_token = visual_token
