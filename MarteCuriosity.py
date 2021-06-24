# -*- coding: utf-8 -*-
"""
Created on Sat May 22 11:42:26 2021

@author: Victor
"""
from sqlalchemy import create_engine            

import pandas as pd
import numpy as np
import streamlit as st

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

st.sidebar.title('Menu')
paginaSelecionada  = st.sidebar.selectbox('Selecione a página', ['Projeto','Relatório Climático Marte', 'Regressão Linear Múltipla'])

if paginaSelecionada == 'Projeto':
    st.markdown("<h1 style='text-align: center; color: black;'>Marte - Curiosity Rover (MSL) </h1>", unsafe_allow_html=True) 
    st.write("<p align='justify'> Por Marcos da Silva Correia e Victor Augusto Souza Resende  <p align='justify'>", unsafe_allow_html=True) 
    st.write("<p align='justify'> 24 de Junho de 2021 <p align='justify'>", unsafe_allow_html=True) 
    st.write("<p align='justify'> O rover Curiosity, parte da missão Mars Science Laboratory, pode ser considerada uma dos feitos mais bem sucedidas da raça humana em outro planeta, nesse caso, em Marte. O rover está no planeta vermelho desde 12 de Agosto de 2012 até então, entretanto, para a confecção desse projeto foram utilizados os dados coletados até o dia 07 de Abril de 2021. Então, com a coleta dos dados por parte do robô, foi possível acessá-los em uma API por meio da linguagem de programação Python. Dessa forma, foi possível executar um projeto de ciência de dados e machine learning de ponta à ponta, com aplicações e validações estatísticas sobre os dados coletados pelo rover. <p align='justify'>", unsafe_allow_html=True) 
    st.write("<p align='justify'> Esse projeto foi executado utilizando a metodologia de mineração de dados denominada CRISP-DM. De maneira rápida, o CRISP é considerado um modelo de processo de mineração de dados que descreve abordagens comumente usadas por especialistas em mineração de dados para resolver problemas, desde a criação de perguntas norteadoras às abordagens que serão utilizadas para responder tais perguntas. Dessa forma, nesse projeto, foi solicitado a resposta de duas perguntas norteadoras: <p align='justify'>", unsafe_allow_html=True) 
    st.write("""<p align='justify'>
                 <ul>
                 <li> Quais características climáticas e ambientais Marte apresentou nos anos de exploração? </li>
                 <li> Dada as variáveis, é possível fazer uma previsão em relação à temperatura média máxima do solo de Marte?  </li>
                 </ul> 
                 <p align='justify'>""", unsafe_allow_html=True)
    st.write("<p align='justify'> Então, ao fim da etapa de implementação do CRISP-DM, dedicou-se a criação de uma hospedagem web via streamlit, da qual é uma biblioteca da linguagem Python, para a implementação da conclusão deste projeto. Portanto, no canto superior esquerdo, no menu, é possível acessar as duas conclusões, a primeira em relação ao relatório climático de Marte e a segunda sobre o modelo de regressão linear múltiplo criado, do qual possui interação com o usuário. <p align='justify'>", unsafe_allow_html=True) 
    st.write("<p align='justify'> Caso queira ler a documentação desse projeto na íntegra, da qual conta com mais de 35 páginas, basta visitar o repositório do autor no GitHub, do qual pode ser acessado por meio do seguinte link: <a href='https://github.com/victoresende19/marte'>Curiosity Repositório</a> <p align='justify'>", unsafe_allow_html=True) 
    st.image("marte1.png", width=None)  

elif paginaSelecionada == 'Relatório Climático Marte':
    st.markdown("<h1 style='text-align: center; color: black;'>Relatório Climático Marte</h1>", unsafe_allow_html=True) 
    
    st.write("\n\n")
    
    st.write("<p align='justify'> Após concluir a análise exploratória, dedicou-se a criação final desse relatório a fim de responder a primeira pergunta norteadora. Dessa maneira, esse relatório servirá como resumo e conclusão da análise exploratória feita, trazendo as informações gerais do possível clima meteorológico encontrado na cratera Gale em Marte pelo rover Curiosity. Então, a seguir estão as informações que compõe o relatório climático da cratera Gale em Marte, de acordo com os dados da API com histórico coletado pelo rover Curiosity no período de 07 de Agosto de 2012 a 07 de Abril de 2021. <p align='justify'>", unsafe_allow_html=True) 
    st.write("<p align='justify'> Como explicado em momentos anteriores, é válido ressaltar que o planeta Marte possui uma atmosfera bastante rasa, ou seja, uma camada tênue da qual contêm os gases ali presentes, assim afetando diretamente o clima e ambiente do planeta. Foi realizado, há alguns anos, estudos dos quais sugerem que, em algum momento do passado, Marte poderia ter tido características químicas e climáticas parecidas com a do planeta Terra. Dessa maneira, tentar entender o clima presente no planeta marciano faz-se justo para entender a história do sistema solar do qual os seres humanos se encontram. <p align='justify'>", unsafe_allow_html=True) 
   
    st.image("marte.png", width=None)
   
    st.write("<p align='justify'> Então, levando em consideração todos os dados coletados até então pelo rover Curiosity em sua jornada no planeta vermelho, entende-se que o clima seria majoritariamente frio. Como visto nesse projeto, as temperaturas do solo e do ar são diferentes devido à falta de atmosfera no planeta. Sendo assim, por conta disso, como citado pela CAB: Imagine que você estivesse no equador marciano ao meio-dia, você se sentiria como o verão aos seus pés, mas o inverno na sua cabeça. Portanto, tais temperaturas seja mínimas ou máximas eram diferentes, entretanto, majoritariamente, pode-se considerar as temperaturas frias para um ser humano. <p align='justify'>", unsafe_allow_html=True) 
    st.write("<p align='justify'> É interessante ressaltar que o planeta marciano possui solo arenoso e ambiente desértico, o que pode identificar grande amplitude térmica em relação ao solo, pois ambientes desérticos possuem grande capacidade de absorção do calor de dia, porém quando o sol se põe o clima torna-se frio.Da mesma forma, solos arenosos acabam por terem maior porosidade, havendo um menor contato entre as partículas do solo, dificultando assim o processo de condução do calor. Sendo assim, o ambiente solo em Marte possui grande amplitude térmica diante da falta de atmosfera e por possuir solo arenoso. <p align='justify'>", unsafe_allow_html=True) 
    st.write("<p align='justify'> Em relação à pressão atmosférica, Marte possui uma pressão atmosférica em média 160 vezes menor do que em relação ao planeta Terra com uma média geral de 700 Pascais. Entretanto, como o rover Curiosity está localizado na cratera de Gale, a pressão atmosférica captada pelo dispositivo REMS é maior, em torno de 800 pascais, uma vez que crateras são mais profundas que o solo propriamente dito. Dessa maneira, a pressão atmosférica em Marte é menor do que em relação à do planeta Terra devido ao motivo da atmosfera marciana ser bastante tênue. <p align='justify'>", unsafe_allow_html=True) 
    st.write("<p align='justify'> Ao que diz respeito ao histórico dos dados referentes ao nível de radiação ultravioleta, como explicado anteriormente, o rover Curiosity, por meio do dispositivo REMS interpreta o nível da radiação ultravioleta em 4 categorias. Como analisado, radiações de categorias moderada e alta foram as mais frequentes nos dias em que o rover coletou tais dados. Consequentemente, a falta de uma atmosfera com ozônio no planeta vermelho, faz com que a radiação ultravioleta consiga atingir a superfície marciana, diferente do planeta Terra do qual possui uma atmosfera com ozônio, repelindo a maioria dos respectivos raios ultravioletas. <p align='justify'>", unsafe_allow_html=True) 
    st.write("<p align='justify'> Percebe-se então, que a atmosfera possui um papel muito importante em relação à regulação térmica de um planeta. Sendo assim, a atmosfera também tem o papel de conter o calor irradiado pelo planeta. Da mesma maneira, a atmosfera contem gases dos quais são denominados gases de efeito estufa, que garantem que parte do calor que chega ao planeta fique retido no planeta em questão. <p align='justify'>", unsafe_allow_html=True) 
    st.write("<p align='justify'> Conclui-se que Marte possui a falta de uma atmosfera densa, ou seja, a existência de uma carência de gases, principalmente os de efeito estufa natural. Em conjunto com o solo arenoso, o clima de Marte é afetado diretamente em características como temperatura, pressão e o nível de radiação recebido na superfície marciana. Dessa forma a absorção de calor torna-se pífia, tornando Marte um planeta majoritariamente frio e nocivo para o ser humano. <p align='justify'>", unsafe_allow_html=True) 
    
    st.image("marte2.png", width=None)
    
    
elif paginaSelecionada == 'Regressão Linear Múltipla':
    #CONEXAO SQLITE
    db = create_engine('sqlite:///projetoIntegrador4_Final.sqlite', echo = False)
    conn = db.connect()
    
    #OBTENDO O DATA FRAME DO SQLITE
    query = '''SELECT * FROM curiosity'''
    df = pd.read_sql_query( query, conn )
    df['DataTerra'] = pd.to_datetime(df['DataTerra']).dt.tz_localize(None)
    df = df[(df.DataTerra >= '2012-08-07') & (df.DataTerra <= '2021-04-07')].reset_index()
    
    
    #Temperatura Maxima Novos Intervalos
    #AR
    df = df.loc[(df['MaxTempAr'] > -35) & (df['MaxTempAr'] < 10)]
    #SOLO
    df = df.loc[(df['MaxTempSolo'] > -25) & (df['MaxTempSolo'] < 30)]
    
    #Temperatura Minima Novos Intervalos
    #AR
    df = df.loc[(df['MinTempAr'] > -95) & (df['MinTempAr'] < -55)]
    #SOLO
    df = df.loc[(df['MinTempSolo'] < -55)]
    
    
    #Fazendo transformacao NIVEL UV
    mapping_dictionary = {"NivelUV":{ "Low": 1, "Moderate": 2, "High": 3, "Very_High": 4}}
    df = df.replace(mapping_dictionary)
    
    
    #REGRESSAO MULTIPLA
    X = df[['MaxTempAr', 'MinTempAr', 'NivelUV']]
    Y = df['MaxTempSolo']
    
    model = LinearRegression() #criando a variavel pra usar reg linear
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state=1) #separando os dados para treino e teste
    model.fit(X_train, Y_train)#treinando o modelo
        
    y_test_predicted = model.predict(X_test)
    
    st.markdown("<h1 style='text-align: center; color: black;'>Regressão Linear Múltipla</h1>", unsafe_allow_html=True) 
    st.write("<p align='justify'> Regressão múltipla é uma coleção de técnicas estatísticas para construir modelos que descrevem de maneira razoável relações entre várias variáveis explicativas de um determinado processo. Ou seja, de maneira simples, o modelo de regressão faz a atribuição de um valor contínuo a um elemento. Sendo assim, o modelo estatístico para a regressão múltipla pode ser representado,<b> generalizadamente</b>, pela seguinte formula: <p align='justify'>", unsafe_allow_html=True) 
    st.latex(r'Y=\beta_{0}+\beta_{1}x_{i1}+\beta_{2}x_{i2}+...+\beta_{p}x_{ip}+\epsilon_i,~~~i=1,...,n') 
    
    st.write("""<p align='justify'> <ul> 
         <li> Sendo x valores das variáveis explicativas.</li>
         <li> Sendo β os parâmetros ou coeficientes de regressão. </li> 
         <li> Sendo ϵ o erro aleatório independente.</li>
         </ul> <p align='justify'>""", unsafe_allow_html=True)
    
    st.write('Preencha os dados de cada variável (Basta arrastar a bolinha vermelha de cada nível):')
    
    TemperaturaMaxAr = st.slider('Temperatura Máxima do Ar: ', min_value=-100, max_value=50, value=-35)
    TemperaturaMinAr = st.slider('Temperatura Mínima do Ar: ', min_value=-100, max_value=50, value=-35)
    NivelUV = st.slider('Nível UV: ', min_value=1, max_value=5, value=1)
    
    st.title('Temperatura Máxima do Solo em Marte')
    new_array = np.array([TemperaturaMaxAr, TemperaturaMinAr, NivelUV]).reshape(-1, 3)
    st.write('Após aplicar o modelo de regressão nos dados, a Temperatura Máxima do Solo em Marte é de: {} graus Celsius'.format(model.predict(new_array)))
    
    
    st.title('Métricas Regressão Linear Múltipla') 
    st.write("Raiz do Erro Médio Quadrático - RSME: {}".format(mean_squared_error(Y_test, y_test_predicted, squared = False)))
    st.write("Erro Médio Absoluto - MSA: {}".format(mean_absolute_error(Y_test, y_test_predicted)))
    
    conn.close()
