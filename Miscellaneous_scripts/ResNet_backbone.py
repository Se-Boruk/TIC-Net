import torch
from torchvision import models

# Pobieramy model i wagi
model = models.resnet50(weights='DEFAULT')

# Zapisujemy wagi do pliku w Twoim folderze projektowym
torch.save(model.state_dict(), "Models/Pretrained/resnet50_weights.pth")
print("Wagi zapisane pomyślnie. Możesz teraz odciąć internet.")