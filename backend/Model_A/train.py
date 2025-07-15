import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm  # <-- progress bar

def main():
    # =====================================
    # CONFIGURATION
    # =====================================
    checkpoint_path = r"C:/Users/anmol/OneDrive/Desktop/DeepDetect/checkpoints/best_model_3way.pth"
    test_dir = r"C:/Users/anmol/OneDrive/Desktop/Hackathon/DeepDetect/Datasets/Test"
    class_names = ['ai_generated', 'deepfake', 'real']   # Match your folder names

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # =====================================
    # TRANSFORMS
    # =====================================
    transform = transforms.Compose([
        transforms.Resize((380,380)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    ])

    # =====================================
    # DATASET & DATALOADER
    # =====================================
    dataset = datasets.ImageFolder(test_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4
    )

    print("Class to index mapping:", dataset.class_to_idx)

    # =====================================
    # MODEL
    # =====================================
    weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1
    model = models.efficientnet_b4(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 3)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    # =====================================
    # EVALUATION
    # =====================================
    all_preds = []
    all_labels = []

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating")
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # =====================================
    # METRICS
    # =====================================
    print("\nClassification Report:\n")
    print(classification_report(
        all_labels, all_preds, target_names=class_names, digits=4
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == "__main__":
    main()
