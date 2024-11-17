import torch
import pandas as pd
from torchvision import datasets, transforms
from model import CustomCNN

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data loading and transformation
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dataset = datasets.ImageFolder(root='data/test', transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

# Load trained model
model = CustomCNN(num_classes=4).to(device)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Predictions
results = []
with torch.no_grad():
    for inputs, filenames in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)

        for fname, pred in zip(filenames, predicted):
            results.append({'Filename': fname, 'Predicted_Label': pred.item()})

# Save predictions
df_results = pd.DataFrame(results)
df_results.to_csv('custom_CNN.csv', index=False, header= False)
print('Predictions saved to custom_CNN.csv')
