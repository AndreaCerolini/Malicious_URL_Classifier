import os
import numpy as np
import pandas as pd
import pickle
import unicodedata
from sklearn.preprocessing import LabelEncoder

PATH = 'Malicious URL v3.csv' 

# SAVE_PATH = "/Users/andreacerolini/Desktop/"

# Percorso della cartella corrente 

SAVE_PATH = os.path.join(os.getcwd(), "preprocessed_data")

# Crea la cartella se non esiste

os.makedirs(SAVE_PATH, exist_ok=True)


# ----------------------------- CARICAMENTO DATI -----------------------------

df = pd.read_csv(PATH).to_numpy()     

# Rimuovo la prima colonna perchè nel dataset è un indice inutile

df = np.delete(df, 0, axis=1)                 

# Coverto in array di stringhe sia gli URL che le label

X = df[:,0].astype(str)                       

y = df[:,1].astype(str)                     

# ------------------------ NORMALIZZAZIONE UNICODE ---------------------------

# Normalizziamo gli URL in forma Unicode compatibile (NFKC): Normalization Form Compatibility Composition

def norm(s):

    s = unicodedata.normalize("NFKC", str(s))
    
    # Inoltre sostituiamo gli spazi non standard (NBSP) con spazi normali.

    return s.replace("\u00A0", " ")

# Quindi applichiamo la normalizzazione a tutti gli URL

X = np.array([norm(s) for s in X])

# ------------------------- ENCODING DELLE LABEL -----------------------------

# Utilizziamo un LabelEncoder per convertire le etichette testuali in interi 0..C-1

le = LabelEncoder()

y = le.fit_transform(y)                        

# ----------------------------- VOCABOLARIO ----------------------------------

# Costruiamo il set dei caratteri presenti negli URL, riservando 0=PAD (riempimento) e 1=UNK (caratteri non visti)

PAD, UNK = 0, 1

# La funzione sorted ordina i caratteri rendendoli un set 

char_set = sorted({char for string in X for char in string})  

# Mappiamo ogni carattere a un indice intero, iniziando da 2 per lasciare spazio a PAD e UNK

string_to_index = {char: i+2 for i, char in enumerate(char_set)} 

# Calcoliamo la dimensione del vocabolario includendo PAD e UNK

vocab_size = len(string_to_index) + 2                        

print(f"Primi 20 caratteri del vocabolario (dim:{vocab_size}):\n",
      dict(list(string_to_index.items())[:20]),"\n")

# ----------------------- ENCODING A SEQUENZA DI INDICI ----------------------

# Fissiamo una lunghezza massima per gli URL in quanto quelli benevoli tendono ad essere più corti
# di conseguenza verranno troncati o riempiti con PAD, mentre quelli di altri tipi (es phishing) sono più lunghi

MAX_LEN = 250

# Funzione per convertire una stringa in una sequenza di indici interi

def encode(string):

    # Mappiamo ogni carattere a un indice intero, iniziando da 2 per lasciare spazio a PAD e UNK

    # .get(char, UNK) restituisce l'indice del carattere o UNK se non è presente
    # di ogni stringa di cui prendiamo solo i primi MAX_LEN caratteri (quindi tronchiamo se più lunga)

    idx = [string_to_index.get(char, UNK) for char in string[:MAX_LEN]]  

    # Se la stringa è più corta di MAX_LEN, aggiungiamo PAD fino a raggiungere la lunghezza desiderata

    if len(idx) < MAX_LEN:

        idx += [PAD] * (MAX_LEN - len(idx))

    return idx

# Adesso ho quindi una matrice contentente gli URL codificati come sequenze di indici interi
# la matrice avrà shape (Numero URL, MAX_LEN).
# I numeri li salvo come int32 in quanto nn.Embedding richiede interi e non float. Non ha senso usare int64
# perchè gli indici non saranno mai così grandi.

X_idx = np.array([encode(string) for string in X], dtype=np.int64)       

print(f"Esempi di X_idx (shape={X_idx.shape}):")

print(X_idx[:3],"\n")

# ------------------------------- SALVATAGGI ---------------------------------

np.save(f"{SAVE_PATH}/X.npy", X_idx)           

np.save(f"{SAVE_PATH}/y.npy", y)

# Salviamo il mapping dei caratteri e i metadati utili in un file pickle

with open(f"{SAVE_PATH}/string_to_index.pkl", "wb") as f:

    pickle.dump({

        "string_to_index": string_to_index,
        
        "PAD": PAD,

        "UNK": UNK,

        "vocab_size": vocab_size,

        "MAX_LEN": MAX_LEN,

        "labels": list(le.classes_)
    }, f)

print("Salvato in:", SAVE_PATH,"\n")

print("File scritti: X.npy, y.npy, string_to_index.pkl","\n")

print("X shape:", X_idx.shape, "| y shape:", y.shape,)