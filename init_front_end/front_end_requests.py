import requests
import time

stopper = time.time()
response = requests.get("http://34.134.225.55:8080/search",params={'query':'vote'})
print(len(response.json()))
print(response.json())
stopper = time.time() - stopper
print(stopper)

# stopper = time.time()
# response = requests.get("http://34.134.225.55/search", params={'query':'data science'})
# stopper = stopper - time.time()
# print(len(response.json()))
# print(response.json())
# print(stopper)

#
# response = requests.post("http://127.0.0.1:8080/search_anchor",params={'query':'how to make pasta'})
# print(len(response.json()))
# print(response.json())

