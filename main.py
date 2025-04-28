import streamlit as st
import pandas as pd
import plotly.graph_objects as go
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

# Prepara os dados para o gráfico
max_tamanho = max(df["tamanho"].max(), tamanho_inteiro if tamanho > 0 else 0)
tamanho_range = pd.Series(range(int(df["tamanho"].min()), int(max_tamanho) + 1))
preco_range = modelo.predict(tamanho_range.values.reshape(-1, 1))

# Cria o gráfico interativo com Plotly
fig = go.Figure()

# Adiciona pontos reais
fig.add_trace(go.Scatter(x=df["tamanho"], y=df["preco"], mode='markers', name='Dados Reais', marker=dict(color='blue')))

# Adiciona linha de regressão
fig.add_trace(go.Scatter(x=tamanho_range, y=preco_range.flatten(), mode='lines', name='Linha de Regressão', line=dict(color='red')))

# Se o usuário digitou um tamanho > 0, faz a previsão e marca o ponto
if tamanho > 0:
    preco_previsto = modelo.predict([[tamanho]])[0][0]
    st.write(f"O valor estimado para um bolo de tamanho {tamanho_inteiro} é **R$ {preco_previsto:.2f}**")
    
    # Adiciona o ponto previsto
    fig.add_trace(go.Scatter(x=[tamanho_inteiro], y=[preco_previsto], mode='markers', name='Preço Previsto', marker=dict(color='green', size=12, symbol='x')))

# Personaliza o layout do gráfico
fig.update_layout(
    title='Relação entre o Tamanho e o Preço do Bolo',
    xaxis_title='Tamanho do Bolo (cm)',
    yaxis_title='Preço do Bolo (R$)',
    legend_title='Legenda',
    template='plotly_white',
    autosize=True
)

# Exibe o gráfico no Streamlit
st.plotly_chart(fig, use_container_width=True)
