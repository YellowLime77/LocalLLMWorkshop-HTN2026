# Run "pip install openai" first

from llm import chat
print(chat([{"role": "user", "content": "Write a haiku about Hack the North"}]))