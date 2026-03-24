import random
import csv
from plot_functions import  mostrar_grafico, plotar_fronteira, plotar_dataset, plotar_matriz_confusao
from helper import carregar_dataset, dividir_treino_teste_estratificado

X = []
y = []


class Perceptron:
    def __init__(self, n_inputs, learning_rate=0.18):
        self.learning_rate = learning_rate
        self.weights = [random.uniform(-1, 0), random.uniform(0, 1)]
        # self.bias = 0.0
        
    def internal_product(self, inputs):
        activation_potential = 0
        for i in range(len(inputs)):
            activation_potential += self.weights[i] * inputs[i]
        # activation_potential += self.bias
        return activation_potential

    def activation_function(self, linear_output):
        if linear_output >= 0:
            return 1
        else:
            return -1
    
    def predict(self, inputs):
        linear_output = self.internal_product(inputs)
        output = self.activation_function(linear_output)
        return output
    
    def train(self, inputs, label):
        predicted_output = self.predict(inputs)
        error = label - predicted_output
        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate * error * inputs[i]
        # self.bias += self.learning_rate * error

X, y, indices = carregar_dataset("salmon_seabass.csv")
# X, y, indices = carregar_dataset("salmon_seabass_moredata.csv")
X_treino, y_treino, idx_treino, X_teste, y_teste, idx_teste = dividir_treino_teste_estratificado(X, y, indices)
perceptron = Perceptron(n_inputs=2, learning_rate=0.1)

epochs = 20
for epoch in range(epochs):
    dados = list(zip(X_treino, y_treino))
    random.shuffle(dados)
    for entradas, label in dados:
        perceptron.train(entradas, label)


print("Pesos finais aprendidos:", perceptron.weights)
mostrar_grafico(X, y, perceptron)

acerto_do_modelo = 0
modelo_acertou_negativo = 0
modelo_errou_positivo = 0
modelo_errou_negativo = 0

for i in range(len(X_teste)):
    pred = perceptron.predict(X_teste[i])
    real = y_teste[i]
    print("Index:", idx_teste[i])
    print("Entrada:", X_teste[i])
    print("Esperado:", real)
    print("Previsto:", pred)
    print("-----")

    if pred == 1 and real == 1:
        acerto_do_modelo += 1
    elif pred == -1 and real == -1:
        modelo_acertou_negativo += 1
    elif pred == 1 and real == -1:
            modelo_errou_positivo += 1
    elif pred == -1 and real == 1:
        modelo_errou_negativo += 1

accuracy = (acerto_do_modelo + modelo_acertou_negativo) / (acerto_do_modelo + modelo_acertou_negativo + modelo_errou_positivo + modelo_errou_negativo)
precision = acerto_do_modelo / (acerto_do_modelo + modelo_errou_positivo) if (acerto_do_modelo + modelo_errou_positivo) > 0 else 0
recall = acerto_do_modelo / (acerto_do_modelo + modelo_errou_negativo) if (acerto_do_modelo + modelo_errou_negativo) > 0 else 0

print("\nMatriz de Confusão")
print("acerto_do_modelo:", acerto_do_modelo)
print("modelo_acertou_negativo:", modelo_acertou_negativo)
print("modelo_errou_positivo:", modelo_errou_positivo)
print("modelo_errou_negativo:", modelo_errou_negativo)

print("\nMétricas")
print("Acurácia:", accuracy)
print("Precisão:", precision)
print("Recall:", recall)

plotar_matriz_confusao(acerto_do_modelo, modelo_acertou_negativo, modelo_errou_positivo, modelo_errou_negativo)