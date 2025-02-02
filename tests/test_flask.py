import requests

data_input = {
    'input_text' : 'this is a product phrase'
}

headers = {
    'Content-Type' : 'application/json'
}

r = requests.post(
    "http://localhost:5000/get_recommendation", 
    headers=headers, 
    json=data_input
)

print(r.json())
print(r.status_code, r.reason)