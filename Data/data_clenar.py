import os
import shutil

# Input per la directory di origine
origine = input("Inserisci il percorso della directory di origine (premere Invio per utilizzare la directory corrente): ")
if not origine:
    origine = os.getcwd()

# Input per la directory di destinazione
destinazione = input("Inserisci il percorso della directory di destinazione (premere Invio per utilizzare la directory corrente): ")
if not destinazione:
    destinazione = os.getcwd()

# Crea la nuova cartella se non esiste già
new_folder_name = "Dati_puliti"
new_folder_path = os.path.join(destinazione, new_folder_name)

if not os.path.exists(new_folder_path):
    os.makedirs(new_folder_path)

counter = 1
    
# Trova tutte le sotto-cartelle nella directory di origine
for folder in os.listdir(origine):
    if os.path.isdir(os.path.join(origine, folder)):
        # Itera attraverso tutte le sotto-cartelle della cartella principale
        for folder_2 in os.listdir(os.path.join(origine, folder)):
            folder_2_path = os.path.join(origine, folder, folder_2)
            if os.path.isdir(folder_2_path):
                # Percorso della nuova cartella nella directory di destinazione
                new_subfolder_path = os.path.join(new_folder_path, folder_2)

                # Crea la cartella nella nuova cartella se non esiste già
                if not os.path.exists(new_subfolder_path):
                    os.mkdir(new_subfolder_path)
                    print(f"La cartella {new_subfolder_path} è stata creata.")

                # Trova tutte le sotto-cartelle della terza cartella
                for folder_3 in os.listdir(folder_2_path):
                    folder_3_path = os.path.join(folder_2_path, folder_3)
                    if os.path.isdir(folder_3_path):
                        # Contatore per i file copiati

                        # Copia i file dalla cartella folder_3 nella nuova cartella di destinazione corrispondente
                        for file_name in os.listdir(folder_3_path):
                            #if file_name.lower().endswith('res.png') and file_name.startswith('NR_RX_'):
                            file_path = os.path.join(folder_3_path, file_name)
                            #new_file_name = f"{origine}_{folder_2}_{counter}.png"
                            new_file_path = os.path.join(new_subfolder_path, file_name)
                            shutil.copy(file_path, new_file_path)
                            counter += 1

print("Operazione completata.")
