import os
import pandas as pd
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import download_images, parse_string
from constants import entity_unit_map


class EntityExtractor(nn.Module):
    def __init__(self, num_classes):
        super(EntityExtractor, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)


def preprocess_data(csv_file, image_folder):
    df = pd.read_csv(csv_file)
    download_images(df['image_link'].tolist(), image_folder)
    return df


def create_dataset(df, image_folder, transform):
    images = []
    labels = []
    for _, row in df.iterrows():
        img_path = os.path.join(image_folder, os.path.basename(row['image_link']))
        img = Image.open(img_path).convert('RGB')
        img = transform(img)
        images.append(img)

        if 'entity_value' in df.columns:
            value, unit = parse_string(row['entity_value'])
            labels.append(torch.tensor([value, list(entity_unit_map[row['entity_name']]).index(unit)]))

    return torch.stack(images), torch.stack(labels) if labels else None


def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")


def predict(model, test_loader, test_df):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    predictions = []
    with torch.no_grad():
        for inputs in tqdm(test_loader, desc="Predicting"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            predictions.extend(outputs.cpu().numpy())

    results = []
    for pred, (_, row) in zip(predictions, test_df.iterrows()):
        value, unit_index = pred
        unit = list(entity_unit_map[row['entity_name']])[int(unit_index)]
        results.append(f"{value:.2f} {unit}")

    return results


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_file = os.path.join(base_dir, 'dataset', 'train.csv')
    test_file = os.path.join(base_dir, 'dataset', 'test.csv')
    image_folder = os.path.join(base_dir, 'images')
    output_file = os.path.join(base_dir, 'test_out.csv')

    if not os.path.exists(image_folder):
        os.makedirs(image_folder)

    # Preprocess data
    print("Preprocessing train data...")
    train_df = preprocess_data(train_file, image_folder)
    print("Preprocessing test data...")
    test_df = preprocess_data(test_file, image_folder)

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Create datasets
    print("Creating datasets...")
    train_images, train_labels = create_dataset(train_df, image_folder, transform)
    test_images, _ = create_dataset(test_df, image_folder, transform)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(list(zip(train_images, train_labels)), batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_images, batch_size=32)

    # Initialize model
    num_classes = 2  # value and unit
    model = EntityExtractor(num_classes)

    print("Training model...")
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    train_model(model, train_loader, criterion, optimizer)

    print("Making predictions...")
    predictions = predict(model, test_loader, test_df)

    output_df = pd.DataFrame({
        'index': test_df['index'],
        'prediction': predictions
    })
    output_df.to_csv(output_file, index=False)
    print(f"Predictions saved to {output_file}")


if __name__ == "__main__":
    main()