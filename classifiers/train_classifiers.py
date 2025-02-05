import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, get_scheduler
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os

MIXED_SIZE = 500
DATASET_PATH = f'path/to/dataset' # this can be original, method generated, simple generated, generated + original all models are trained in the same way  

RUBRICS = [
    'Statement of what should be proven: A proof by contraposition of an implication consists in showing that if x rational, then x^2 is rational. ',
    'Correct assumption: x is rational [Assumption] ',
    'Correct proof reasoning',
    'Proof conclusion: By contraposition, if x^2 is irrational, then x is irrational.'
]

RUBRIC_HYPERPARAMS = {
    1: {'batch_size': 16, 'epochs': 14},
    2: {'batch_size': 16, 'epochs': 14},
    3: {'batch_size': 16, 'epochs': 14},
    4: {'batch_size': 16, 'epochs': 14},
}

LEARNING_RATE = 2e-5
SEED = 2024
SAVE_DIR = './saved_models'  
os.makedirs(SAVE_DIR, exist_ok=True)  


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)

# train-test splits
# this script assumes that the original data was split into train-test sets and this split is saved under 'SAVE_SPLIT_PATH'
SAVE_SPLIT_PATH = './saved_splits/'   
os.makedirs(SAVE_SPLIT_PATH, exist_ok=True)  

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def readData(path, is_csv=True):
    """
    Reads data from a specified path. Supports both Excel and CSV formats.
    For CSV, expects 'text', 'rubric1', 'rubric2', 'rubric3', 'rubric4' columns.
    """
    # for generated datasets
    if is_csv:
        df = pd.read_csv(path)
        df = df.rename(columns={
            "text": "CONTRAPOSITION task",
            "rubric1": RUBRICS[0],
            "rubric2": RUBRICS[1],
            "rubric3": RUBRICS[2],
            "rubric4": RUBRICS[3]
        })
    # for the original dataset
    else:
        df = pd.read_excel(path, sheet_name='contraposition')
        df = df[['CONTRAPOSITION task'] + RUBRICS].copy()
    
    df = df.dropna()  
    return df


def tokenize_data(tokenizer, texts, labels):
    """
    Tokenizes the text and returns a TensorDataset.
    """
    encodings = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
    dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'], torch.tensor(labels))
    return dataset

def train_model(model, train_loader, val_loader, optimizer, scheduler, epochs, device):
    """
    Trains the model and evaluates it on the test set at the end of training.
    """
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * input_ids.size(0)
        print(f"Epoch {epoch + 1} Loss: {total_loss / len(train_loader):.4f}")

    # Evaluation
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, label = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())
            labels.extend(label.cpu().numpy())
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    kappa = cohen_kappa_score(labels, preds)  
    return accuracy, f1, kappa


def save_split(train_df, val_df):
    """
    Saves the train-test split as CSV files.
    """
    train_df.to_csv(os.path.join(SAVE_SPLIT_PATH, 'train_split.csv'), index=False)
    val_df.to_csv(os.path.join(SAVE_SPLIT_PATH, 'val_split.csv'), index=False)
    print(f"Train-test split saved to {SAVE_SPLIT_PATH}")


def load_split():
    """
    Loads the train-test split from CSV files.
    """
    # this is the original dataset 
    train_df = pd.read_csv(os.path.join(SAVE_SPLIT_PATH, 'train_split.csv'))
    val_df = pd.read_csv(os.path.join(SAVE_SPLIT_PATH, 'val_split.csv'))
    print(f"Train-test split loaded from {SAVE_SPLIT_PATH}")
    return train_df, val_df


# Training and Evaluation Pipeline
def main():

    train_df = readData(DATASET_PATH, is_csv=True)

    _, val_df = load_split()

    results = pd.DataFrame(columns=["Rubric", "Kappa", "Accuracy", "F1"])
    
    # For each rubric, process and train
    for rubric_idx, rubric_column in enumerate(RUBRICS, start=1):
        print(f"\nTraining Rubric {rubric_idx}: {rubric_column}")

        batch_size = RUBRIC_HYPERPARAMS[rubric_idx]['batch_size']
        epochs = RUBRIC_HYPERPARAMS[rubric_idx]['epochs']

        train_dataset = tokenize_data(tokenizer, train_df['CONTRAPOSITION task'].tolist(), train_df[f'{rubric_column}'].tolist())
        val_dataset = tokenize_data(tokenizer, val_df['CONTRAPOSITION task'].tolist(), val_df[f'{rubric_column}'].tolist())

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01, eps = 1e-8)
        num_training_steps = len(train_loader) * epochs
        scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0.1 * num_training_steps, num_training_steps=num_training_steps)

        # Train and evaluate
        accuracy, f1, kappa = train_model(model, train_loader, val_loader, optimizer, scheduler, epochs, device)
        print(f"Rubric {rubric_idx} - Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Cohen's Kappa: {kappa:.4f}")
        new_row = pd.DataFrame([{
            "Rubric": rubric_idx,
            "Kappa": kappa,
            "Accuracy": accuracy,
            "F1": f1
        }])

        results = pd.concat([results, new_row], ignore_index=True)
        # Save the model
        model_save_path = os.path.join(SAVE_DIR, f"rubric_{rubric_idx}_model.pt")
        torch.save(model.state_dict(), model_save_path)

    results.to_csv(f"{SAVE_DIR}/test_results/results.csv", index=False)


if __name__ == "__main__":
    main()
