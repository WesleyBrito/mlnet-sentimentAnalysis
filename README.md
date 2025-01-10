# Sentiment Analysis using ML.NET

Este projeto utiliza o **ML.NET** para realizar análise de sentimentos em dados de texto, como avaliações de usuários. O modelo é treinado com um conjunto de dados e pode ser usado para prever se um sentimento é positivo ou negativo.

## Funcionalidades
- **Carregamento de Dados:** Carrega e divide os dados em conjuntos de treinamento e teste.
- **Treinamento do Modelo:** Utiliza regressão logística estocástica (SDCA) para treinar o modelo.
- **Avaliação do Modelo:** Mede métricas como acurácia, AUC e F1-Score.
- **Predições em Lote e Individuais:** Realiza previsões em lote ou para entradas fornecidas pelo usuário em tempo real.

## Estrutura do Código

### Principais Componentes

1. **Carregamento dos Dados**
   ```csharp
   TrainTestData LoadData(MLContext mLContext, string dataPath)
   ```
   - Carrega os dados de um arquivo de texto e os divide em conjuntos de treinamento e teste (80/20).

2. **Treinamento do Modelo**
   ```csharp
   ITransformer BuildAndTrainModel(MLContext mLContext, IDataView splitTrainSet)
   ```
   - Constrói e treina um modelo usando o algoritmo SDCA Logistic Regression.

3. **Avaliação do Modelo**
   ```csharp
   void EvaluateModel(MLContext mLContext, ITransformer model, IDataView splitTestSet)
   ```
   - Avalia a qualidade do modelo usando métricas como:
     - **Accuracy:** Proporção de predições corretas.
     - **AUC:** Capacidade de separar classes positivas e negativas.
     - **F1-Score:** Balanceia precisão e revocação.

4. **Predição de Sentimentos**
   ```csharp
   SentimentPrediction UseModelWithSingleItem(MLContext mLContext, ITransformer model, string sentiment)
   ```
   - Permite ao usuário prever o sentimento de uma entrada de texto.

5. **Visualização de Dados**
   ```csharp
   void ViewData(TrainTestData splitDataView, int take)
   ```
   - Exibe exemplos do conjunto de dados de treinamento.

6. **Loop de Predição**
   ```csharp
   void RunPredictionLoop(MLContext mLContext, ITransformer model)
   ```
   - Permite ao usuário inserir textos para prever sentimentos até que "bye" seja digitado.

## Pré-requisitos
- **.NET SDK:** Certifique-se de ter o .NET SDK instalado. Você pode baixá-lo [aqui](https://dotnet.microsoft.com/download).
- **Arquivo de Dados:** O arquivo `yelp_labelled.txt` contendo os dados de treinamento deve estar na pasta `Data` do projeto.
- **Este projeto está preparado para rodar em `devcontainer` se preferir.**

## Como Executar

1. Clone este repositório.
   ```bash
   git clone <url-do-repositorio>
   ```

2. Navegue até a pasta do projeto.
   ```bash
   cd <pasta-do-projeto>
   ```

3. Execute o programa.
   ```bash
   dotnet run
   ```

4. Insira frases para prever os sentimentos, ou digite "bye" para sair.

## Conjunto de Dados

O arquivo de dados `yelp_labelled.txt` deve seguir o seguinte formato:
```
Texto do Sentimento<TAB>Label
```
- **Label:** `1` para positivo e `0` para negativo.

Exemplo:
```
This is an amazing product!    1
The service was terrible.      0
```

## Saída
- Durante a execução, o programa exibirá:
  - Métricas do modelo.
  - Resultados de predições para as entradas fornecidas pelo usuário.

## Tecnologias Utilizadas
- **ML.NET:** Biblioteca para aprendizado de máquina em .NET.
- **C#:** Linguagem de programação usada no projeto.

## SdcaLogisticRegression
O SdcaLogisticRegression é um treinador de modelo no ML.NET que utiliza o algoritmo SDCA (Stochastic Dual Coordinate Ascent) para resolver problemas de classificação binária, como prever se um sentimento é positivo ou negativo. Ele combina a eficiência do SDCA com a função de perda logística, que é ideal para tarefas de classificação.

Como funciona o SdcaLogisticRegression?
O que é SDCA?

É um algoritmo de otimização iterativo que ajusta os parâmetros do modelo em pequenos lotes de dados, reduzindo o custo computacional e permitindo a escalabilidade.
Trabalha no espaço dual, usando variáveis duals para resolver o problema de otimização de forma eficiente.
Regressão Logística:

A regressão logística utiliza a função sigmoide para mapear as previsões em um intervalo de 0 a 1, permitindo interpretar as saídas como probabilidades.
A saída é usada para classificar os exemplos em uma das duas categorias (por exemplo, "positivo" ou "negativo").
Combinação SDCA + Regressão Logística:

O SDCA ajusta os pesos do modelo para minimizar a perda logística, garantindo que o modelo seja treinado de maneira eficiente, mesmo com grandes conjuntos de dados.
Vantagens do SdcaLogisticRegression
Escalabilidade: Funciona bem com grandes conjuntos de dados devido à abordagem estocástica.
Eficiência: Atualiza os pesos do modelo iterativamente, garantindo convergência rápida.
Interpretação Probabilística: As previsões retornam probabilidades, facilitando a interpretação dos resultados.

## Contato
Caso tenha dúvidas ou sugestões, entre em contato pelo email: `wesley.brito.oliveira@gmail.com`.
