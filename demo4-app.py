import os
import api_keys

from groq import Groq

client = Groq(
    api_key=os.environ(api_keys.groq_api)
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "return me today's date in india",
        }
    ],
    model="llama3-8b-8192",
)

print(chat_completion.choices[0].message.content)