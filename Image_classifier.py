import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn.functional as F
from torch.utils.data import random_split
from Model import *

# Preparazione dei dati
transform = transforms.Compose([
    #transforms.Resize((256, 256)),
    #transforms.Resize((200, 200)), 
    transforms.Resize((128, 128)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_path = 'Data/train_Dati_256_256/'

#train_dataset = datasets.ImageFolder(train_data_path, transform=transform)
#validation_dataset = datasets.ImageFolder(validation_data_path, transform=transform)
#
#train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
#validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)


# Caricamento del dataset completo
full_dataset = datasets.ImageFolder(data_path, transform=transform)

# Suddivisione del dataset in training set e validation set
train_size = int(0.8 * len(full_dataset))  # 80% del dataset per il training
val_size = len(full_dataset) - train_size  # Il restante 20% per la validazione
train_dataset, validation_dataset = random_split(full_dataset, [train_size, val_size])

# Creazione dei data loader per il training set e il validation set
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)


# Definizione del modello
#model = CNNTransformerModel(num_classes=len(full_dataset.classes))  #CNN + Trasformer
#model = ResNet64Model(num_classes=len(full_dataset.classes))   # Trasformer + CNN
model = ResNet32Model(num_classes=len(full_dataset.classes))   # Trasformer + CNN

# Utilizzo delle GPU se disponibili
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")
model.to(device)

# Definizione della funzione di perdita e dell'ottimizzatore
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


model_name=model.name+".pth"
model_save_path = "Models/"+model_name

# Addestramento del modello
num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_dataset)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total

    
    torch.save(model.state_dict(), model_save_path)

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")




with open("Results/"+model.name+".txt", "a") as file:
    # Scrivi le informazioni desiderate nel file
    file.write("Informazioni sul modello")
    file.write("Modello: {}\n".format(model.name))
    file.write("\n")
