import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn.functional as F
from torch.utils.data import random_split


# Definizione del blocco residuo
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

# Definizione della rete ResNet-32
class ResNet32(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet32, self).__init__()
        self.in_channels = 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self.make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 128, num_blocks[2], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 128)

    def make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        #out = F.relu(self.bn1(self.conv1(x)))
        out = torch.sigmoid(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class ResNet64(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet64, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self.make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self.make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self.make_layer(block, 128, num_blocks[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 128)

    def make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        #out = torch.sigmoid(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# Funzione per la creazione di una ResNet-32
def resnet32(num_classes=10):
    return ResNet32(ResidualBlock, [5, 5, 5], num_classes=num_classes)  # 3 blocchi con 5 residui ciascuno

def resnet64(num_classes=10):
    return ResNet64(ResidualBlock, [3, 4, 6, 3], num_classes=num_classes)


class CNNTransformerModel(nn.Module):
    def __init__(self, num_classes):
        self.name="CNNTransformerModel"
        super(CNNTransformerModel, self).__init__()
        self.cnn= resnet64(num_classes=num_classes)
        #self.dropout = nn.Dropout(p=0.5) # l ho messo per provare cosa accade
        # Rimuovi gli strati completamente connessi di ResNet
        self.cnn.fc = nn.Identity()
        # Aggiungi un trasformatore per l'elaborazione delle feature
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8,batch_first=True)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=3)
        # Strati completamente connessi per la classificazione
        self.fc1 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Estrai le feature con la CNN
        features = self.cnn(x)
        # Ridimensiona le feature per adattarle al trasformatore
        features = features.view(features.size(0), features.size(1), -1)
        features = features.permute(2, 0, 1)  # Cambia l'ordine delle dimensioni
        # Passa le feature attraverso il trasformatore
        features = self.transformer(features)
        # Ridimensiona le feature di output del trasformatore
        features = features.permute(1, 2, 0)  # Cambia nuovamente l'ordine delle dimensioni
        features = features.view(features.size(0), -1)
        # Classificazione finale
        output = self.fc1(features)
        return output
    
    def load(self, model_path, map_location='cpu'):
        # Carica lo stato del modello
        state_dict = torch.load(model_path, map_location=map_location)
        # Carica lo stato del modello nella rete neurale
        self.load_state_dict(state_dict)




class ResNet64Model(nn.Module):
    def __init__(self, num_classes):
        self.name="ResNet64Model"
        super(ResNet64Model, self).__init__()
        self.resnet = resnet64(num_classes=num_classes)
        # Rimuovi gli strati completamente connessi di ResNet
        self.resnet.fc = nn.Identity()
        # Strato completamente connesso per la classificazione
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # Estrai le feature con la ResNet
        features = self.resnet(x)
        # Classificazione finale
        output = self.fc(features)
        return output
    
    def load(self, model_path, map_location='cpu'):
        # Carica lo stato del modello
        state_dict = torch.load(model_path, map_location=map_location)
        # Carica lo stato del modello nella rete neurale
        self.load_state_dict(state_dict)




class ResNet32Model(nn.Module):
    def __init__(self, num_classes):
        self.name="ResNet32Model"
        super(ResNet32Model, self).__init__()
        self.resnet = resnet32(num_classes=num_classes)
        # Rimuovi gli strati completamente connessi di ResNet
        self.resnet.fc = nn.Identity()
        # Strato completamente connesso per la classificazione
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # Estrai le feature con la ResNet
        features = self.resnet(x)
        # Classificazione finale
        output = self.fc(features)
        return output
    
    def load(self, model_path, map_location='cpu'):
        # Carica lo stato del modello
        state_dict = torch.load(model_path, map_location=map_location)
        # Carica lo stato del modello nella rete neurale
        self.load_state_dict(state_dict)





