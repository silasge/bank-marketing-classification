# Objetivo

A Ciência de Dados tem um papel importante no processo de tomada de decisão de uma empresa. A alta disponibilidade de dados e a utilização de técnicas preditivas permitem às empresas otimizar seus custos com uma campanha ao focar nos clientes com maior potencial de rentabilidade.

Neste repositório farei uma aplicação da ciência de dados na área de marketing. Utilizo uma campanha de um banco português que tinha como objetivo fazer com que clientes fizessem depósito a prazo, um tipo de produto bancário onde a instituição guarda os recursos do cliente por um prazo determinado (daí o nome) em troca de pagamento de juros. O objetivo será aplicar modelos de Machine Learning para tentar prever quais clientes irão fazer o depósito.

# Dataset

Os dados são retirados da [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) e são compostos de 21 variáveis contendo atributos sobre o cliente, sobre a campanha e sobre o contexto sócio-econômico. A lista completa de variáveis e suas descrições pode ser encontrada em [*references*](references/bank-additional-names.txt).

# Tecnologias

A aplicação foi toda feita em Python, utilizando o Poetry para gerenciar as dependências e o GNU Make para gerenciar toda a pipeline de leitura, processamento dos dados, treinamento e avaliação dos modelos.

A leitura e manipulação dos dados foram feitas com o Pandas e Numpy. As visualizações dos dados eu utilizei o Altair, uma ferramenta que lembra bastante o ggplot2 do R e aproveitei esse pequeno projeto para aprendê-lo. Os notebooks são, como de costume, feitos com Jupyter Notebooks.

Apliquei quatro modelos nos dados: Regressão Logística, o Support Vector Classifier com o kernel linear (o LinearSVC do sklearn), Árvores de Decisão (Decision Trees) e Florestas Aleatórias (Random Forests). Todos os modelos e otimização dos parâmetros foram feitos com o sklearn.

# Como Rodar o Projeto

O projeto precisa ter o Python instalado, além do [Poetry](https://python-poetry.org/docs/master/) e do [GNU Make](https://www.gnu.org/software/make/).

Com todos instalados, clone este repositório e rode o make com o seguinete comando:

``` bash
make all
```

Ao final ele dará a performance do melhor modelo no conjunto de teste entre todos os avaliados.

Opcionalmente, é possível definir alguns parâmetros para o treinamento dos modelos:

* TEST_SIZE: A proporção da base de dados que é divida para o conjunto de teste. Deve ser uma float entre 0 e 1. O padrão é 0.25;
* RANDOM_STATE: Um número inteiro definindo o random state para replicação dos dados. O padrão é 42.
* CV: O número de folds de cross-validation utilizado em RandomSearchCV. O padrão é 5.
* N_ITER: O número de iterações utilizadas por RandomSearchCV para testar diferentes hiperparâmetros. O padrão 50.
* THRESHOLD: A probabilidade limite a partir do qual atribui uma predição para a classe positiva. O padrão é 0.4.

Que podem ser usados da seguinte forma, modificando um ou mais parâmetros:

``` bash
make all TEST_SIZE=0.2 CV=20 ...
```

# Análises

* [Exploração dos Dados](notebooks/1.0-data-exploration.ipynb)
* [Análise dos Modelos](notebooks/2.0-model-analysis.ipynb)
* Post no Blog (Em construção)

# Contato

* [LinkedIn](https://www.linkedin.com/in/silasge/)
* [Twitter](https://twitter.com/_silasge)
* [sg.lopes26@gmail.com](mailto:sg.lopes26@gmail.com)