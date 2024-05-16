from flask import Flask, request, jsonify, session
from flask_cors import CORS
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)  # For session management
CORS(app, supports_credentials=True)  # Enable Cross-Origin Resource Sharing with credentials
MAX_TOKENS_LIMIT = 4192 # Maximum tokens allowed

# Load the CosmoXL model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("allenai/cosmo-xl")
model = AutoModelForSeq2SeqLM.from_pretrained("allenai/cosmo-xl")

# Set the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define the system prompt
system_prompt = """
You are an AI assistant named Sera, designed to be an empathetic listener and supportive companion for women who want to rant or vent about various aspects of their lives.
Your role is to provide a safe and non-judgmental space for them to express their feelings, frustrations, and experiences.
Throughout the conversation, maintain a compassionate and supportive tone, and avoid dismissing or minimizing the user's feelings or experiences.
Your goal is to create a safe space for them to vent and receive the kind of support they need, whether that's affirmations or advice. 
Remember, as an AI assistant, you should respond based on the provided information and avoid making assumptions or judgments about the user's personal life or circumstances.
"""

# Function to generate a response
def generate_response(rant, conversation_history):
    input_text = f"{system_prompt} <sep> {' <turn> '.join(conversation_history)} <turn> {rant}"
    inputs = tokenizer([input_text], return_tensors="pt", truncation=True).to(device)
    
    # Check if the input exceeds the maximum token limit
    input_tokens_count = inputs.input_ids.size(1)
    if input_tokens_count > MAX_TOKENS_LIMIT:
        return "Oh, you've reached the token limit. Please restart or refresh the screen to start a new conversation."

    outputs = model.generate(inputs.input_ids, max_length=1024, do_sample=True, top_p=0.95, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Route to handle user input and generate response
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('input')
    if 'conversation_history' not in session:
        session['conversation_history'] = []

    conversation_history = session['conversation_history']

    # Generate response from the AI model
    response = generate_response(user_input, conversation_history)

    # Update conversation history
    conversation_history.append(f"User: {user_input}")
    conversation_history.append(f"Sera: {response}")
    session['conversation_history'] = conversation_history

    return jsonify({'response': response, 'conversation_history': conversation_history})

# Route to reset the conversation
@app.route('/reset', methods=['POST'])
def reset():
    session.pop('conversation_history', None)
    return jsonify({'message': 'Conversation history has been reset.'})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
