import os
import shutil
import random

def divide_e_copia(directory_origine, directory_destinazione):
    # Ottieni il nome della directory di origine
    origine_name = os.path.basename(os.path.abspath(directory_origine))

    print("Directory di origine:", directory_origine)

    
    # Crea le cartelle per i dataset di training e validation
    train_dir = os.path.join(directory_destinazione, "train_" + str(origine_name))
    validation_dir = os.path.join(directory_destinazione, "validation_" + str(origine_name))

    print("Cartella di addestramento:", train_dir)
    print("Cartella di validazione:", validation_dir)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(validation_dir, exist_ok=True)

    # Percorre tutte le cartelle e sottocartelle nella directory di origine
    for root, dirs, files in os.walk(directory_origine):
        # Crea una struttura di cartelle simile nella directory di destinazione
        rel_path = os.path.relpath(root, directory_origine)
        new_dir_train = os.path.join(train_dir, rel_path)
        new_dir_validation = os.path.join(validation_dir, rel_path)
        os.makedirs(new_dir_train, exist_ok=True)
        os.makedirs(new_dir_validation, exist_ok=True)

        # Divide casualmente i file in due gruppi (70% training, 30% validation)
        files_train = random.sample(files, int(len(files) * 0.7))
        files_validation = [file for file in files if file not in files_train]

        # Copia i file nel set di addestramento nella nuova directory di destinazione
        for file_name in files_train:
            file_path_origine = os.path.join(root, file_name)
            file_path_destinazione = os.path.join(new_dir_train, file_name)
            shutil.copy(file_path_origine, file_path_destinazione)

        # Copia i file nel set di validation nella nuova directory di destinazione
        for file_name in files_validation:
            file_path_origine = os.path.join(root, file_name)
            file_path_destinazione = os.path.join(new_dir_validation, file_name)
            shutil.copy(file_path_origine, file_path_destinazione)

# Input per la directory di origine
origine = input("Inserisci il percorso della directory di origine: ")

# Input per la directory di destinazione
destinazione = input("Inserisci il percorso della directory di destinazione: ")

# Effettua la divisione e la copia
divide_e_copia(origine, destinazione)

print("Operazione completata.")
