from groq import Groq
import os

class GROQ:
    def __init__(self, system, GroqAPI_path, conversation_path=None):
        self.system = system

        # Load all the conversations
        self.prompt = self.__load_conversation__(conversation_path) if conversation_path else []
        # Load Groq's API key
        self.__load_client__(GroqAPI_path)

    def __load_client__(self, GroqAPI_path):
        with open(GroqAPI_path, "r", encoding="utf-8") as f:
            GROQ_API_KEY = str(f.read().strip())

        self.client = Groq(api_key = GROQ_API_KEY)

    def generate(self, text, model_name = "llama3-70b-8192"):
        text = "Human: " + text
        self.prompt.append(text)

        completion = self.client.chat.completions.create(
            messages = [
                {
                    "role": "system",
                    "content": self.system
                },
                {
                    "role": "user",
                    "content": "\n".join(self.prompt) + "\nWINTER: "
                }
            ],
            model = model_name
        )

        response = completion.choices[0].message.content
        self.prompt.append("WINTER: " + response)
        return response
    
    def __load_conversation__(self, path):
        if os.path.isfile(path) == False:
            return []

        with open(path, "r", encoding="utf-8") as f:
            return [i.strip() for i in f.readlines()]

    # Save all the messages, conversations and prompts
    def save_conversation(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.prompt) + "\n")
