from flask import Flask, render_template, request, jsonify
from rag_engine import get_rag_response as get_response

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    user_question = data.get("question", "")
    reply = get_response(user_question)
    return jsonify({"answer": reply})

if __name__ == "__main__":
    app.run(debug=True)
