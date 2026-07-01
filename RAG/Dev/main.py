from dotenv import load_dotenv
from langchain_core import __version__ as core_version 
from langgraph.version import __version__ as lg_version
from groq import Groq

load_dotenv()

print(f"Langchain-core version: {core_version}")
print(f"Langgraph version: {lg_version}")

def main():
    client = Groq()
    chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Say 'Setup complete!' in one word.",
        }
    ],
    model="llama-3.1-8b-instant",
    )

    print(chat_completion.choices[0].message.content)

if __name__ == "__main__":
    main()
