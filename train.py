import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from data_fetcher import StockDataFetcher, GramianAngularField
from model import GAFCNN


class StockDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def prepare_data(ticker='AAPL', lookback=60, image_size=64):
    fetcher = StockDataFetcher(ticker)
    df = fetcher.fetch()

    gaf = GramianAngularField(method='summation', image_size=image_size)

    images = []
    labels = []

    for i in range(len(df) - lookback - 1):
        window = df['close'].iloc[i:i+lookback].values

        gaf_image = gaf.transform(window)

        future_price = df['close'].iloc[i+lookback+1]
        current_price = df['close'].iloc[i+lookback]

        label = 1 if future_price > current_price else 0

        images.append(gaf_image)
        labels.append(label)

    return np.array(images), np.array(labels)


def train_model(num_epochs=10, batch_size=32, learning_rate=0.001):
    print("Preparing data...")
    images, labels = prepare_data('AAPL', lookback=60, image_size=64)

    images = torch.FloatTensor(images).unsqueeze(1)
    labels = torch.LongTensor(labels)

    dataset = StockDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = GAFCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch_images, batch_labels in dataloader:
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    return model


if __name__ == "__main__":
    model = train_model()
    torch.save(model.state_dict(), 'trained_model.pth')
    print("Training complete! Model saved.")
