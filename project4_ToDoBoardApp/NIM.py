import os
from dotenv import load_dotenv
from openai import OpenAI

# loading API keys from env
load_dotenv()
NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")

print(f"LOG NVIDIA_API_KEY{NVIDIA_API_KEY}")
if NVIDIA_API_KEY == '':
    raise ValueError("Please add your own API key in the .env file")


class NIM:
    def __init__(self):
        print("LOG NIM initialized..")
        #  api_key = "$API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC"
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=NVIDIA_API_KEY
        )
        self.plan_prompt = ""
        with open('PlanPrompt.txt', 'r') as file:
            self.PLAN_PROMPT = file.read()

    def generate_plan_prompt(self, history, user_prompt):
        self.plan_prompt = self.PLAN_PROMPT.replace("HISTORY", str(history))
        return  self.plan_prompt + user_prompt

    def get_plan_response(self, prompt):
        completion = self.client.chat.completions.create(
            model="meta/llama-3.1-405b-instruct",
            messages=[{"role":"user","content": prompt}],
            temperature=0.2,
            top_p=0.7,
            max_tokens=1024,
            stream=True
        )

        plan_completion = ""
        for chunk in completion:
            if chunk.choices[0].delta.content is not None:
                plan_completion += chunk.choices[0].delta.content

        print("LOG NIM get_plan_response plan_completion ", plan_completion)
        return plan_completion
