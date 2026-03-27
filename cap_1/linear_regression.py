import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Carregar e preparar os dados
data_root = "https://github.com/ageron/data/raw/main/"
lifesat = pd.read_csv(data_root + "lifesat/lifesat.csv")
x = lifesat[["GDP per capita (USD)"]].values
y = lifesat[["Life satisfaction"]].values
paises = ["Argentina", "Brasil", "Chipre", "Paraguai"]
previsoes = []

# Visualizar os dados
lifesat.plot(kind='scatter', grid=True, x="GDP per capita (USD)", y="Life satisfaction")
plt.axis([23_500, 62_500, 4, 9])
plt.show()

# Selecionar modelo linear
model = LinearRegression()

# Treinar o modelo
model.fit(x, y)

# Fazer uma predicao para a Argentina
x_new = [[14_359]] # PIP per capita da Argentina em 2025 (Previsao do FMI)
pred = model.predict(x_new)[0][0]
previsoes.append(pred)

# Fazer uma predicao para o Brasil
x_new = [[10_578]] # PIP per capita do Brasil em 2025 (Previsao do FMI)
pred = model.predict(x_new)[0][0]
previsoes.append(pred)

# Fazer uma predicao para o Chipre
x_new = [[37_655.2]] # PIP per capita do Chipre em 2020
pred = model.predict(x_new)[0][0]
previsoes.append(pred)

# Fazer uma predicao para o Paraguai
x_new = [[6_799]] # PIP per capita do Paraguai em 2025 (Previsao do FMI)
pred = model.predict(x_new)[0][0]
previsoes.append(pred)

# Grafico de barra
plt.figure()
plt.bar(paises, previsoes)
plt.title("Satisfação de vida prevista por país")
plt.xlabel("País")
plt.xlabel("Satisfação de vida")
plt.show()