import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pytorch_lightning as pl
from Model import *



# Preparazione dei dati
transform = transforms.Compose([
    #transforms.Resize((256, 256)),
    transforms.Resize((200, 200)), 
    #transforms.Resize((128, 128)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_path = 'Data/train_Dati_256_256/'

# Definizione di un callback per il monitoraggio del modello e il salvataggio del checkpoint
checkpoint_callback = pl.callbacks.ModelCheckpoint(
    monitor='val_loss',  # Metrica di validazione da monitorare
    dirpath='model_checkpoints',  # Directory dove salvare i checkpoint
    filename='best_model_pythorch_lightning',  # Nome del file del checkpoint
    save_top_k=1,  # Numero di migliori modelli da salvare
    mode='min',  # Modalità di salvataggio (minimizzare la metrica di validazione)
    save_last=True  # Salva l'ultimo modello alla fine dell'addestramento
)


# Caricamento del dataset completo
full_dataset = datasets.ImageFolder(data_path, transform=transform)

# Suddivisione del dataset in training set e validation set
train_size = int(0.8 * len(full_dataset))  # 80% del dataset per il training
val_size = len(full_dataset) - train_size  # Il restante 20% per la validazione
train_dataset, validation_dataset = random_split(full_dataset, [train_size, val_size])

# Creazione dei data loader per il training set e il validation set
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=16, shuffle=False)

# Definizione di un callback per il monitoraggio del modello
checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss')

# Creazione di un oggetto Lightning Trainer
trainer = pl.Trainer(max_epochs=10, gpus=1, checkpoint_callback=checkpoint_callback)

# Creazione del modello
model = CNNTransformerModel(num_classes=len(full_dataset.classes))

# Addestramento del modello
trainer.fit(model, train_dataloader=train_loader, val_dataloaders=validation_loader)


