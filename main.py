import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def load_law_data(folder_path):
    law_texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):  # Make sure your files are .txt format
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                law_texts.append(file.read())
    return "\n".join(law_texts)

# Path to your folder where law data files are stored
folder_path = "law_data"
law_data = load_law_data(folder_path)

# Load pre-trained model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Set the model in evaluation mode (important for inference)
model.eval()

def generate_response(input_text, max_length=100):
    # Tokenize the input text
    inputs = tokenizer.encode(input_text, return_tensors="pt")
    
    # Generate the response
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1)
    
    # Decode the generated tokens to text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

# Function to interact with the user in the console
def interactive_chat():
    print("Hello! Ask me anything about Indian law.")
    print("Type 'exit' to end the conversation.")
    
    while True:
        # Take input from the user
        user_input = input("\nYour question: ")
        
        # Exit the loop if the user types 'exit'
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        # Generate and display the response from the AI model
        response = generate_response(user_input)
        print("AI Response: ", response)

# Start the interactive chat
interactive_chat()
