import requests



response = requests.post("http://127.0.0.1:8080/search_title",params={"query":"python"})

print(response.json())