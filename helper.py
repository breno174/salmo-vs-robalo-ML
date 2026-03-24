import csv
import random

def carregar_dataset(caminho):
    X = []
    y = []
    indices = []
    with open(caminho) as file:
        reader = csv.DictReader(file)
        for row in reader:
            indice = int(row["index"])
            lightness = float(row["lightness"])
            width = float(row["width"])
            label = int(row["species"])
            if label == 0:
                label = -1
            else:
                label = 1
            X.append([lightness, width])
            y.append(label)
            indices.append(indice)
    return X, y, indices

def dividir_treino_teste_estratificado(X, y, indices, proporcao_treino=0.73):
    
    classe_negativa = []
    classe_positiva = []

    for i in range(len(X)):
        if y[i] == -1:
            classe_negativa.append((X[i], y[i], indices[i]))
        else:
            classe_positiva.append((X[i], y[i], indices[i]))

    random.shuffle(classe_negativa)
    random.shuffle(classe_positiva)

    tamanho_treino_neg = int(len(classe_negativa) * proporcao_treino)
    tamanho_treino_pos = int(len(classe_positiva) * proporcao_treino)

    treino = (
        classe_negativa[:tamanho_treino_neg] +
        classe_positiva[:tamanho_treino_pos]
    )

    teste = (
        classe_negativa[tamanho_treino_neg:] +
        classe_positiva[tamanho_treino_pos:]
    )

    random.shuffle(treino)
    random.shuffle(teste)

    X_treino = [d[0] for d in treino]
    y_treino = [d[1] for d in treino]
    idx_treino = [d[2] for d in treino]

    X_teste = [d[0] for d in teste]
    y_teste = [d[1] for d in teste]
    idx_teste = [d[2] for d in teste]

    return X_treino, y_treino, idx_treino, X_teste, y_teste, idx_teste