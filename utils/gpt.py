import os
import yaml
import base64
import mimetypes
import requests
from dotenv import load_dotenv
from utils.util import read

# Create a .env file in the project root directory and add your Anthropic API key as ANTHROPIC_API_KEY=<your_key>
load_dotenv(dotenv_path=os.path.join("..", ".env"))
api_key = os.getenv("OPENAI_API_KEY")


class Session:
    def __init__(self, model, prompts_file) -> None:
        self.model = model
        self.past_tasks: list[str] = []
        self.past_messages = []
        self.past_responses: list[str] = []

        # Load the predefined prompts for the LLM
        with open(f"{prompts_file}.yaml") as file:
            self.predefined_prompts: dict[str, str] = yaml.safe_load(file)

    def send(self, task: str, prompt_info: dict[str, str] | None = None, images: list[str] = [], file_path=None) -> str:
        print(f"$ --- Sending task: {task}")
        self.past_tasks.append(task)
        prompt = self._make_prompt(task, prompt_info)
        self._send(prompt, images, file_path)
        response = self.past_responses[-1]
        print(f"$ --- Response:\n{response}\n")

        return response

    def _make_prompt(self, task: str, prompt_info: dict[str, str] | None) -> str:
        # Get the predefined prompt for the task
        prompt = self.predefined_prompts[task]

        # Check for task-specific required information ()
        # All tasks that require extra information should have a case here
        valid = True
        match task:
            case "expand_text_prompt":
                valid = "text_prompt" in prompt_info
        if not valid:
            raise ValueError(f"Extra information is required for the task: {task}")

        # Replace the placeholders in the prompt with the information
        if prompt_info is not None:
            for key in prompt_info:
                prompt = prompt.replace(f"<{key.upper()}>", prompt_info[key])

        return prompt

    def _send(self, prompt: str, images: list[str] = [], file_path=None) -> str:
        payload = self._create_payload(prompt, images=images)
        if not os.path.exists(file_path):
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            print("Waiting for LLM to be generated")
            # response = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload, timeout=(5, 120))  # Anthropic
            # response = requests.post("https://aium.cc/v1/chat/completions", headers=headers, json=payload, timeout=(5, 120)) #If you are in mainland China, you can try aium.cc
            response = requests.post("https://api.gptsapi.net/v1/chat/completions", headers=headers, json=payload,
                                     timeout=(5, 120))  # WildCard
            print(f"LLM response: {response.text}")
            try:
                response = response.json()['choices'][0]['message']['content']
            except:
                print(f"$ --- Error Response: {response.json()}\n")
        else:
            response = read(file_path)
        self.past_messages.append({"role": "assistant", "content": response})
        self.past_responses.append(response)

    def _create_payload(self, prompt: str, images: list[str] = []):
        """Creates the payload for the API request."""
        messages = {
            "role": "user",
            "content": [],
        }

        for image in images:
            base64_image = encode_image(image)
            image_content = {
                ## ChatGPT
                # "type": "image_url",
                # "image_url": {
                #     "url": base64_image,
                #     "detail": "auto",
                # }
                # Claude
                'type': 'image',
                'source': {
                    'type': 'base64',
                    'media_type': 'image/png',
                    'data': base64_image
                }
            }
            messages["content"].append(image_content)

        messages["content"].append({
            "type": "text",
            "text": prompt,
        })

        self.past_messages.append(messages)

        return {
            "model": self.model,
            "system": self.predefined_prompts["system"],
            "messages": self.past_messages,
        }


def encode_image(image_path: str):
    """Encodes an image to base64 and determines the correct MIME type."""
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        raise ValueError(f"Cannot determine MIME type for {image_path}")

    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        # return f"data:{mime_type};base64,{encoded_string}"  # ChatGPT
        return encoded_string  # Claude
