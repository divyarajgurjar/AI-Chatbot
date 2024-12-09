from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

# Define the prompt template
template = """
Answer the question below
Here is the conversation history: {context}

Question: {question}

Answer:
"""

# Initialize the model and chain
model = OllamaLLM(model="llama3")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def handle_conversation():
    """
    Handle the chatbot conversation in a command-line interface.
    """
    context = ""  # Initialize conversation context
    print("Welcome to the AI Chatbot! Type 'exit' to quit.")

    while True:
        # Take user input
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        try:
            # Generate response
            result = chain.invoke({"context": context, "question": user_input})
            print("Bot:", result)
            # Update context
            context += f"\nUser: {user_input}\nAI: {result}"
        except Exception as e:
            print(f"Error: {e}")
            print("Bot: Sorry, I encountered an error. Please try again.")

if __name__ == "__main__":
    handle_conversation()
