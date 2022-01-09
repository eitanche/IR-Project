import requests



# response = requests.get("http://127.0.0.1:8080/search_title",params={'query':'python'})
# print(len(response.json()))
# print(response.json())

response = requests.get("http://127.0.0.1:8080/search_anchor",params={'query':'data science'})
print(len(response.json()))
print(response.json())
#
# response = requests.post("http://127.0.0.1:8080/search_anchor",params={'query':'how to make pasta'})
# print(len(response.json()))
# print(response.json())

