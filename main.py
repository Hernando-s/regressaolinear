import streamlit as st
import pandas as pd 
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Lê o arquivo CSV
df = pd.read_csv("bolos.csv")

# Remove espaços extras nas colunas
df.columns = df.columns.str.strip()

# Cria e treina o modelo
modelo = LinearRegression()
x = df[["tamanho"]]  # Certifique-se que "tamanho" é uma coluna válida
y = df[["preco"]]    # Certifique-se que "preco" é uma coluna válida
modelo.fit(x, y)

# Interface Streamlit
st.title("Previsão do Valor do Bolo")
st.divider()

# Entrada do usuário
tamanho = st.number_input("Digite o tamanho do bolo", min_value=0, step=1)  # Tamanho inteiro
tamanho_inteiro = int(tamanho)  # Converte para inteiro

# Verifica se o valor foi inserido
if tamanho > 0:
    preco_previsto = modelo.predict([[tamanho]])[0][0]
    st.write(f"O valor estimado para um bolo de tamanho {tamanho_inteiro} é R$ {preco_previsto:.2f}")

# Exibe o gráfico
# Criar o gráfico com os dados reais e a linha de previsão
fig, ax = plt.subplots()

# Plotando os dados reais (Tamanho vs Preço)
ax.scatter(df["tamanho"], df["preco"], color='blue', label='Dados Reais')

# Plotando a linha de regressão
tamanho_range = pd.Series([t for t in range(int(df["tamanho"].min()), int(df["tamanho"].max())+1)], name="tamanho")
preco_range = modelo.predict(tamanho_range.values.reshape(-1, 1))
ax.plot(tamanho_range, preco_range, color='red', label='Linha de Regressão')

# Adicionando rótulos e título
ax.set_xlabel('Tamanho do Bolo (cm)')
ax.set_ylabel('Preço do Bolo (R$)')
ax.set_title('Relação entre o Tamanho e o Preço do Bolo')
ax.legend()
ax.grid(True)

# Exibe o gráfico no Streamlit
st.pyplot(fig)
#python -m streamlit run main.py