import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data import Field, LabelField, BucketIterator

# Define the fields for preprocessing
TEXT = Field(tokenize='spacy', lower=True)
LABEL = LabelField(dtype=torch.float)

# Load the IMDB dataset
train_data, test_data = IMDB.splits(TEXT, LABEL)

# Build the vocabulary
TEXT.build_vocab(train_data, max_size=10000, vectors="glove.6B.100d")
LABEL.build_vocab(train_data)

# Create iterators for the data
batch_size = 64
train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
)

# Define the model architecture
class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.rnn(embedded)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        return self.fc(hidden)

# Initialize the model, optimizer, and loss function
vocab_size = len(TEXT.vocab)
embedding_dim = 100
hidden_dim = 256
output_dim = 1
dropout = 0.5

model = SentimentClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, dropout)
optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss()

# Move model to device (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
criterion = criterion.to(device)

# Training loop
def train(model, iterator, optimizer, criterion):
    model.train()
    for batch in iterator:
        optimizer.zero_grad()
        predictions = model(batch.text).squeeze(1)
        loss = criterion(predictions, batch.label)
        loss.backward()
        optimizer.step()

# Evaluate function
def evaluate(model, iterator, criterion):
    model.eval()
    with torch.no_grad():
        total_loss = 0
        total_correct = 0
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            loss = criterion(predictions, batch.label)
            total_loss += loss.item()
            predictions = torch.round(torch.sigmoid(predictions))
            total_correct += (predictions == batch.label).sum().item()
    return total_loss / len(iterator), total_correct / len(iterator.dataset)

# Train the model
num_epochs = 5
for epoch in range(num_epochs):
    train(model, train_iterator, optimizer, criterion)
    train_loss, train_acc = evaluate(model, train_iterator, criterion)
    valid_loss, valid_acc = evaluate(model, test_iterator, criterion)
    print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.3f}, Val. Loss: {valid_loss:.3f}, Val. Acc: {valid_acc:.3f}')
