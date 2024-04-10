import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from Model import *  # Assicurati di importare correttamente la tua classe CNNTransformerModel

# Disabilita l'utilizzo del modulo HPU
os.environ["TORCH_HPU_PRELOAD"] = "0"

# Definizione del dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#inputs, labels = inputs.to(device), labels.to(device)


# Preparazione dei dati di validation
transform = transforms.Compose([
    #transforms.Resize((256, 256)),  # Ridimensiona le immagini a 64x64
    #transforms.Resize((200, 200)),
    transforms.Resize((128, 128)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

validation_data_path = 'Data/validation_Dati_256_256/'
validation_dataset = datasets.ImageFolder(validation_data_path, transform=transform)
validation_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

# Definizione della funzione di perdita
criterion = nn.CrossEntropyLoss()

# Creazione di un'istanza del modello
#model = CNNTransformerModel(len(validation_dataset.classes)).to(device)  # Assicurati che la classe CNNTransformerModel sia definita correttamente
#model =ResNet64Model(num_classes=len(validation_dataset.classes))   
model = ResNet32Model(num_classes=len(validation_dataset.classes))   # Trasformer + CNN
model.to(device)

model_save_path = "Models/ResNet32Model.pth"

# Carica lo stato del modello

model.load(model_save_path, device)

# Imposta il modello in modalità valutazione
model.eval()
# Validazione del modello
correct = 0
total = 0
class_correct = [0] * len(validation_dataset.classes)
class_total = [0] * len(validation_dataset.classes)

with torch.no_grad():
    for inputs, labels in validation_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)  # Utilizza il modello per ottenere le predizioni
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Calcola l'accuratezza per ciascuna classe
        for i in range(len(labels)):
            label = labels[i]
            class_correct[label] += (predicted[i] == label).item()
            class_total[label] += 1

# Stampa l'accuratezza globale
with open("Results/"+model.name+".txt", "a") as file:        

    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy*100:.2f}%\n")
    file.write(f"Validation Accuracy: {accuracy*100:.2f}%\n")


    # Calcola e stampa l'accuratezza per classe
    file.write("Accuracy for classes\n")
    for i in range(len(validation_dataset.classes)):
        accuracy = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
        print(f'Accuracy of class {validation_dataset.classes[i]}: {accuracy:.2f}%')
        file.write(f'Accuracy of class {validation_dataset.classes[i]}: {accuracy:.2f}%\n')
