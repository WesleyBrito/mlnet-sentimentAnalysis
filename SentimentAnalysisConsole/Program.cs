using Microsoft.ML;
using Microsoft.ML.Data;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using static Microsoft.ML.DataOperationsCatalog;
using static System.Console;

#region const
const string dataFileName = "yelp_labelled.txt";
const string folderName = "Data";
const int ViewDataQuantity = 10;
#endregion

Stopwatch watch = new Stopwatch();

string _dataPath = Path.Combine(Environment.CurrentDirectory, folderName, dataFileName);

MLContext mLContext = new MLContext(seed: 0);

mLContext.Log += (s, e) => WriteLine(e.Message);

TrainTestData splitDataView = LoadData(mLContext, _dataPath);

ViewData(splitDataView, ViewDataQuantity);

ITransformer model = BuildAndTrainModel(mLContext, splitDataView.TrainSet);

EvaluateModel(mLContext, model, splitDataView.TestSet);

const string exitPhase = "bye";
while (true)
{
    Console.ForegroundColor = ConsoleColor.White;
    WriteLine();
    WriteLine("Enter a sentiment to predict (or 'bye' to exit):");
    string sentiment = ReadLine();
    if (sentiment == exitPhase)
    {
        break;
    }
    SentimentPrediction resultPrediction = UseModelWithSingleItem(mLContext, model, sentiment);
    PrintPredictionResult(resultPrediction);
}

/// <summary>
/// Load data from text file
/// </summary>
/// <param name="mLContext">Context from MachineLearning</param>
/// <param name="dataPath">path from file with data to load in the MLContext</param>
/// <returns>A Train Test Data with 20% dedicated to test</returns>
TrainTestData LoadData(MLContext mLContext, string dataPath)
{
    IDataView dataView = mLContext.Data.LoadFromTextFile<SentimentData>(dataPath, hasHeader: false);
    TrainTestData splitDataView = mLContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
    return splitDataView;
}

/// <summary>
/// Build and train the model
/// </summary>
/// <param name="mLContext">Context from MachineLearning</param>
/// <param name="splitTrainSet">Train data</param>
/// <returns>A trained model</returns>
ITransformer BuildAndTrainModel(MLContext mLContext, IDataView splitTrainSet)
{
    var estimator = mLContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.SentimentText))
        .Append(mLContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

    watch.Reset();
    watch.Start();

    ITransformer model = estimator.Fit(splitTrainSet);

    watch.Stop();
    WriteLine($"Training time: {watch.ElapsedMilliseconds} ms");

    return model;
}

/// <summary>
/// Evaluate the model
/// </summary>
/// <param name="mLContext">Context from MachineLearning</param>
/// <param name="model">Trained model</param>
/// <param name="splitTestSet">Test data</param>
/// <returns></returns>
void EvaluateModel(MLContext mLContext, ITransformer model, IDataView splitTestSet)
{
    Console.ForegroundColor = ConsoleColor.DarkCyan;
    WriteLine("Start evaluating model");
    var predictions = model.Transform(splitTestSet);
    var metrics = mLContext.BinaryClassification.Evaluate(predictions, "Label");

    WriteLine();
    WriteLine("Model quality metrics evaluation");
    WriteLine("--------------------------------");

    WriteLine($"Accuracy: {metrics.Accuracy:P2}");
    WriteLine($"Auc: {metrics.AreaUnderRocCurve:P2}");
    WriteLine($"F1Score: {metrics.F1Score:P2}");
    WriteLine("Confusion Matrix");
    WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());

    WriteLine("Finished training model");
}

/// <summary>
/// Use the model with a single item
/// </summary>
/// <param name="mLContext">Context from MachineLearning</param>
/// <param name="model">Trained model</param>
/// <param name="sentiment">Sentiment to predict</param>
/// <returns></returns>
SentimentPrediction UseModelWithSingleItem(MLContext mLContext, ITransformer model, string sentiment)
{
    PredictionEngine<SentimentData, SentimentPrediction> predictionFunction = mLContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

    SentimentData InputSchema = new SentimentData
    {
        SentimentText = sentiment
    };

    return predictionFunction.Predict(InputSchema);
}

/// <summary>
/// Print the prediction result
/// </summary>
/// <param name="sentimentPrediction">Prediction result</param>
/// <returns></returns>
void PrintPredictionResult(SentimentPrediction sentimentPrediction)
{
    Console.ForegroundColor = ConsoleColor.DarkGreen;
    WriteLine();
    WriteLine("=============== Prediction Test of model ===============");

    WriteLine();
    WriteLine($"Sentiment: {sentimentPrediction.SentimentText} | Prediction: {(Convert.ToBoolean(sentimentPrediction.Prediction) ? "Positive" : "Negative")} | Probability: {sentimentPrediction.Probability} ");
    
    WriteLine();
    WriteLine("=============== End of Predictions ===============");
}

/// <summary>
/// View data
/// </summary>
/// <param name="splitDataView">Data to view</param>
/// <param name="take">Number of items to view</param>
/// <returns></returns>
void ViewData(TrainTestData splitDataView, int take)
{
    Console.ForegroundColor = ConsoleColor.DarkYellow;
    WriteLine($"View data: (take {take})");
    foreach (var item in splitDataView.TrainSet.Preview().RowView.Take(take))
    {
        WriteLine($"{item.Values[1].Key}: {item.Values[1].Value}| {item.Values[0].Key}: {item.Values[0].Value}");
    }
    WriteLine("Finish View data");
}