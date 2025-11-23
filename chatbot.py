import json

with open("faq.json", "r") as f:
    faq = json.load(f)

def get_response(user_input):
    user_input = user_input.lower()
    for question, answer in faq.items():
        if question in user_input:
            return answer
    return "I’m not sure about that yet. Please ask something else!"
