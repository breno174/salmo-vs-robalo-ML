import matplotlib.pyplot as plt

def plotar_dataset(X, y):
    x1_classe1 = []
    x2_classe1 = []
    x1_classe2 = []
    x2_classe2 = []

    for i in range(len(X)):
        if y[i] == 1:
            x1_classe1.append(X[i][0])
            x2_classe1.append(X[i][1])
        else:
            x1_classe2.append(X[i][0])
            x2_classe2.append(X[i][1])
    plt.scatter(x1_classe1, x2_classe1, label="Robalo")
    plt.scatter(x1_classe2, x2_classe2, label="Salmão")

def plotar_fronteira(perceptron):
    w1 = perceptron.weights[0]
    w2 = perceptron.weights[1]
    if w2 == 0:
        return
    x_vals = [-10, 10]
    y_vals = [-(w1/w2)*x for x in x_vals]
    plt.plot(x_vals, y_vals, label="fronteira decisão")

def mostrar_grafico(X, y, perceptron):

    plotar_dataset(X, y)

    plotar_fronteira(perceptron)

    plt.xlabel("lightness")
    plt.ylabel("width")

    plt.legend()

    plt.show()

def plotar_matriz_confusao(acerto_do_modelo, modelo_acertou_negativo, modelo_errou_positivo, modelo_errou_negativo):

    matriz = [
        [modelo_acertou_negativo, modelo_errou_positivo],
        [modelo_errou_negativo, acerto_do_modelo]
    ]

    fig, ax = plt.subplots()

    ax.imshow(matriz)

    ax.set_xticks([0,1])
    ax.set_yticks([0,1])

    ax.set_xticklabels(["Pred negativo", "Pred posirtivo"])
    ax.set_yticklabels(["Real negativo", "Real positivo"])

    ax.set_title("Matriz de Confusão")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, matriz[i][j], ha="center", va="center", color="white")

    plt.show()