import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence

# 1. Tokenization and Vocabulary Building
tokenizer = get_tokenizer('basic_english')

def yield_tokens(data_iter):
    for label, text in data_iter:
        yield tokenizer(text)

train_iter = IMDB(split='train')
vocab = build_vocab_from_iterator(yield_tokens(train_iter),
                                  specials=["<unk>"],
                                  max_tokens=25000)
vocab.set_default_index(vocab["<unk>"])

# 2. Pipelines
text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: 1 if x == 'pos' else 0

# 3. Collate Function
def collate_batch(batch):
    label_list, text_list, lengths = [], [], []
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        lengths.append(len(processed_text))
    lengths = torch.tensor(lengths, dtype=torch.int64)
    labels = torch.tensor(label_list, dtype=torch.float32)
    text_padded = pad_sequence(text_list, padding_value=0, batch_first=True)
    return text_padded, labels, lengths

# 4. DataLoaders
batch_size = 64

# Re-instantiate the iterators
train_iter = IMDB(split='train')
test_iter = IMDB(split='test')

# Convert iterators to lists (acceptable for small datasets)
train_list = list(train_iter)
test_list = list(test_iter)

train_dataloader = DataLoader(train_list, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_list, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

# 5. Model Definition
class TextClassificationModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, padding_idx):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, text, lengths):
        embedded = self.embedding(text)  # [batch_size, seq_len, embed_dim]
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        out = self.dropout(hidden[-1])  # Get the last layer's hidden state
        return self.fc(out)

vocab_size = len(vocab)
embed_dim = 100
hidden_dim = 256
output_dim = 1
padding_idx = vocab["<unk>"]

model = TextClassificationModel(vocab_size, embed_dim, hidden_dim, output_dim, padding_idx)

# 6. Loss and Optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 7. Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion.to(device)

# 8. Accuracy Function
def binary_accuracy(preds, y):
    probs = torch.sigmoid(preds).squeeze()
    rounded_preds = torch.round(probs)
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc

# 9. Training and Evaluation Functions
def train(model, dataloader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_acc = 0

    for text, labels, lengths in dataloader:
        text, labels, lengths = text.to(device), labels.to(device), lengths.to(device)

        optimizer.zero_grad()
        predictions = model(text, lengths).squeeze(1)
        loss = criterion(predictions, labels)
        acc = binary_accuracy(predictions, labels)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)

def evaluate(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0

    with torch.no_grad():
        for text, labels, lengths in dataloader:
            text, labels, lengths = text.to(device), labels.to(device), lengths.to(device)
            predictions = model(text, lengths).squeeze(1)
            loss = criterion(predictions, labels)
            acc = binary_accuracy(predictions, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    return epoch_loss / len(dataloader), epoch_acc / len(dataloader)

# 10. Training Loop
num_epochs = 5

for epoch in range(num_epochs):
    train_loss, train_acc = train(model, train_dataloader, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, test_dataloader, criterion)

    print(f'Epoch: {epoch+1}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
