import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers

# Load the law data
def load_law_data(folder_path):
    law_texts = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):  # Ensure only .txt files are loaded
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                law_texts.append(file.read())
    return law_texts

# Path to your folder where law data files are stored
folder_path = r"C:/Users/priya/Documents/Indian_Law_AI/law_data"
law_data = load_law_data(folder_path)

# Create a tokenizer from scratch
def train_tokenizer(law_data):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"])
    
    # Train the tokenizer on your law data
    tokenizer.train_from_iterator(law_data, trainer=trainer)
    
    # Save the trained tokenizer
    tokenizer.save("custom_tokenizer.json")
    return tokenizer

# Train tokenizer
tokenizer = train_tokenizer(law_data)

# Create a dataset using Hugging Face's Dataset library
class LawDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        # Use tokenizer to get the token ids, and extract only the input_ids
        tokenized_text = self.tokenizer.encode(text, add_special_tokens=True).ids
        # Pad the sequences to the same length (max_length in this case)
        tokenized_text = tokenized_text[:self.max_length]  # truncate to max length
        return torch.tensor(tokenized_text)

# Custom collate function to handle padding
def collate_fn(batch):
    # Pad the sequences to the same length (max_length in this case)
    padded_batch = torch.nn.utils.rnn.pad_sequence([torch.tensor(item) for item in batch], batch_first=True, padding_value=tokenizer.token_to_id("<pad>"))
    return padded_batch

# Tokenize the dataset
dataset = LawDataset(law_data, tokenizer)
train_test_split = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
train_dataset, eval_dataset = train_test_split

# Define a Custom Transformer Model with more layers and better capacity
class CustomTransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6):
        super(CustomTransformerModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        transformer_output = self.transformer(embedded, embedded)
        output = self.fc_out(transformer_output)
        return output

# Define the model with the vocabulary size of your custom tokenizer
vocab_size = len(tokenizer.get_vocab())
model = CustomTransformerModel(vocab_size)

# Send model to GPU if available
if torch.cuda.is_available():
    model.cuda()

# Training configuration
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

# DataLoader for training and evaluation
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
eval_loader = DataLoader(eval_dataset, batch_size=2, collate_fn=collate_fn)

# Training loop with increased epochs and more logging
num_epochs = 20  # Increased number of epochs
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        batch = batch.cuda() if torch.cuda.is_available() else batch
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(batch)
        
        # Calculate loss
        loss = criterion(outputs.view(-1, vocab_size), batch.view(-1))
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Save the trained model
torch.save(model.state_dict(), "custom_law_model.pth")

# Now, create a function to generate responses based on the trained model
def generate_response(input_text, model, tokenizer, max_length=100):
    model.eval()  # Set model to evaluation mode
    
    # Tokenize input text
    tokenized_input = tokenizer.encode(input_text, add_special_tokens=True).ids
    tokenized_input = torch.tensor(tokenized_input).unsqueeze(0)  # Add batch dimension
    
    # Move to GPU if available
    if torch.cuda.is_available():
        tokenized_input = tokenized_input.cuda()
    
    # Generate output tokens
    with torch.no_grad():
        output = model(tokenized_input)
        
    # Get the predicted next token
    predicted_token = output.argmax(dim=-1)
    
    # Decode the predicted tokens back to text
    response = tokenizer.decode(predicted_token[0].tolist(), skip_special_tokens=True)
    
    # Fix the spacing issue by manually adding spaces between tokens
    response = ' '.join(response.split())
    
    return response

# Function to interact with the user in the console
def interactive_chat():
    print("Hello! Ask me anything about Indian law.")
    print("Type 'exit' to end the conversation.")
    
    while True:
        user_input = input("\nYour question: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        response = generate_response(user_input, model, tokenizer)
        print("AI Response: ", response)

# Start the interactive chat
interactive_chat()
