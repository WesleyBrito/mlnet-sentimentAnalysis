using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;
using static System.Console;

#region const
const string dataFileName = "yelp_labelled.txt";
const string folderName = "Data";
#endregion

string _dataPath = Path.Combine(Environment.CurrentDirectory, folderName, dataFileName);

MLContext mLContext = new MLContext(seed: 0);

TrainTestData splitDataView = LoadData(mLContext, _dataPath);

TrainTestData LoadData(MLContext mLContext, string dataPath)
{
    IDataView dataView = mLContext.Data.LoadFromTextFile<SentimentData>(dataPath, hasHeader: false);
    TrainTestData splitDataView = mLContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
    return splitDataView;
}

ReadLine();