import openai

# openai.api_base = "https://api.openai.com/v1" # + v1
openai.api_base = "https://apikeyplus.com/v1" # + v1
# openai.api_key = "API_KEY"
openai.api_key = "sk-JvxKN09JvcNAvDZAC13c35Eb8e11454c8f7e8736F070B47f"

for resp in openai.ChatCompletion.create(
                                    model="gpt-4",
                                    messages=[
                                      {"role": "user", "content": "证明费马大定理"}
                                    ],
                                    # 流式输出
                                    stream = True):
    if 'content' in resp.choices[0].delta:
        print(resp.choices[0].delta.content, end="", flush=True)
