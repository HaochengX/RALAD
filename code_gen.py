import openai

def call_gpt_api(query):
    api_key = input("Please enter your API key: ")  
    openai.api_key = api_key

    response = openai.Completion.create(
        engine="gpt-3.5-turbo",
        prompt=query,
        max_tokens=150
    )
    return response.choices[0].text

# 测试 API 调用
result = call_gpt_api("Translate 'hello world' to French")
print(result)