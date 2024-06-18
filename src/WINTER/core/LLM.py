from groq import Groq
import os

with open("bin\cache\\GroqAPI.txt", "r", encoding="utf-8") as f:
    GROQ_API_KEY = str(f.read().strip())

client = Groq(api_key = GROQ_API_KEY)

class LLM:
    def __init__(self, system):
        self.system = system
        self.prompt = []

    def generate(self, text, model = "llama3-8b-8192"):
        self.prompt.append(text)

        completion = client.chat.completions.create(
            messages = [
                {
                    "role": "system",
                    "content": self.system
                },
                {
                    "role": "user",
                    "content": " ".join(self.prompt)
                }
            ],
            model = "llama3-8b-8192"
        )

        response = completion.choices[0].message.content
        self.prompt.append(response)
        return response
