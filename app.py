from flask import Flask, render_template, request, jsonify
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

app = Flask(__name__)

template = """
Answer the question below
Here is the conversation history: {context}

Question: {question}

Answer:
"""
model = OllamaLLM(model="llama3")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/", methods=["POST"])
def chatbot_response():
    data = request.json
    user_message = data.get("message", "")
    context = data.get("context", "")
    result = chain.invoke({"context": context, "question": user_message})
    return jsonify({"response": result})

if __name__ == "__main__":
    app.run(debug=True)
