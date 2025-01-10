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

## Contato
Caso tenha dúvidas ou sugestões, entre em contato pelo email: `wesley.brito.oliveira@gmail.com`.
