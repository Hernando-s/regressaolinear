import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import mpld3
import streamlit.components.v1 as components
from sklearn.linear_model import LinearRegression

# Lê o arquivo CSV
df = pd.read_csv("bolos.csv")
df.columns = df.columns.str.strip()

# Cria e treina o modelo
modelo = LinearRegression()
x = df[["tamanho"]]
y = df[["preco"]]
modelo.fit(x, y)

# Interface Streamlit
st.title("Previsão do Valor do Bolo")
st.divider()

# Entrada do usuário
tamanho = st.number_input("Digite o tamanho do bolo", min_value=0, step=1)
tamanho_inteiro = int(tamanho)

# Prepara o gráfico
fig, ax = plt.subplots()
ax.scatter(df["tamanho"], df["preco"], color='blue', label='Dados Reais')  # Dados reais

# Linha de regressão ajustada
max_tamanho = max(df["tamanho"].max(), tamanho_inteiro if tamanho > 0 else 0)
tamanho_range = pd.Series(range(int(df["tamanho"].min()), int(max_tamanho) + 1))
preco_range = modelo.predict(tamanho_range.values.reshape(-1, 1))
ax.plot(tamanho_range, preco_range, color='red', label='Linha de Regressão')

# Se o usuário digitou um tamanho > 0, faz a previsão e marca no gráfico
if tamanho > 0:
    preco_previsto = modelo.predict([[tamanho]])[0][0]
    st.write(f"O valor estimado para um bolo de tamanho {tamanho_inteiro} é **R$ {preco_previsto:.2f}**")
    
    # Marca o ponto previsto no gráfico
    ax.scatter(tamanho_inteiro, preco_previsto, color='green', s=100, marker='X', label='Preço Previsto')

# Personaliza o gráfico
ax.set_xlabel('Tamanho do Bolo (cm)')
ax.set_ylabel('Preço do Bolo (R$)')
ax.set_title('Relação entre o Tamanho e o Preço do Bolo')
ax.legend()
ax.grid(True)

# Exibe o gráfico interativo
fig_html = mpld3.fig_to_html(fig)
components.html(fig_html, height=600)
