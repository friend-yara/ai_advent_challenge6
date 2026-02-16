import requests, os, sys

prompt = " ".join(sys.argv[1:]) or "Say hello"

r = requests.post(
    "https://api.openai.com/v1/responses",
    headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
    json={"model": "gpt-4.1", "input": prompt},
)

data = r.json()

if r.status_code != 200:
    print(data)  # покажет ошибку API
else:
    print(data["output"][0]["content"][0]["text"])
