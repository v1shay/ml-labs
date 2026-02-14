# ml_labs/execution/cv_trainer.py

import os
import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

from ml_labs.core.model_result import ModelResult


ARTIFACT_DIR = "artifacts"


def train_cv_model(dataset_path: str) -> ModelResult:

    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(dataset_path, transform=transform)

    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    model.train()

    for epoch in range(2):  # short deterministic training
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluate (quick pass)
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in train_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)

    artifact_path = os.path.join(ARTIFACT_DIR, "cv_model.pt")
    torch.save(model.state_dict(), artifact_path)

    return ModelResult(
        model_name="ResNet18",
        modality="image",
        problem_type="classification",
        metrics={"accuracy": accuracy},
        artifact_path=artifact_path,
        metadata={"num_classes": len(dataset.classes)},
    )
