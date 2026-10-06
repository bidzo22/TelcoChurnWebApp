using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;

var mlContext = new MLContext(seed: 1);

string dataPath = Path.Combine(AppContext.BaseDirectory, "churn_clean.csv");
Console.WriteLine($"Завантажуємо дані: {dataPath}");

IDataView dataView = mlContext.Data.LoadFromTextFile<ModelInput>(
    dataPath,
    hasHeader: true,
    separatorChar: ',');

var split = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2, seed: 1);

Console.WriteLine("Починаємо AutoML-тренування (до 5 хвилин)...");

var experimentSettings = new BinaryExperimentSettings
{
    MaxExperimentTimeInSeconds = 300,
    OptimizingMetric = BinaryClassificationMetric.Accuracy
};

var experiment = mlContext.Auto().CreateBinaryClassificationExperiment(experimentSettings);

ExperimentResult<BinaryClassificationMetrics> experimentResult = experiment.Execute(
    trainData: split.TrainSet,
    labelColumnName: nameof(ModelInput.Churn));

var best = experimentResult.BestRun;
Console.WriteLine();
Console.WriteLine($"Найкращий алгоритм: {best.TrainerName}");
Console.WriteLine($"Validation Accuracy: {best.ValidationMetrics.Accuracy:P2}");

var testMetrics = mlContext.BinaryClassification.Evaluate(
    best.Model.Transform(split.TestSet),
    labelColumnName: nameof(ModelInput.Churn));

Console.WriteLine();
Console.WriteLine("=== Результати на тестовій вибірці ===");
Console.WriteLine($"Accuracy: {testMetrics.Accuracy:P2}");
Console.WriteLine($"AUC:      {testMetrics.AreaUnderRocCurve:P2}");
Console.WriteLine($"F1 Score: {testMetrics.F1Score:P2}");

string modelPath = Path.Combine(AppContext.BaseDirectory, "ChurnModel.zip");
mlContext.Model.Save(best.Model, dataView.Schema, modelPath);

Console.WriteLine();
Console.WriteLine($"Модель збережено: {modelPath}");
Console.WriteLine("Готово! Скопіюй ChurnModel.zip у папку MLModel проєкту TelcoChurnWebApp.");

public class ModelInput
{
    [LoadColumn(0)] public string Gender { get; set; }
    [LoadColumn(1)] public float SeniorCitizen { get; set; }
    [LoadColumn(2)] public string Partner { get; set; }
    [LoadColumn(3)] public string Dependents { get; set; }
    [LoadColumn(4)] public float Tenure { get; set; }
    [LoadColumn(5)] public string PhoneService { get; set; }
    [LoadColumn(6)] public string MultipleLines { get; set; }
    [LoadColumn(7)] public string InternetService { get; set; }
    [LoadColumn(8)] public string OnlineSecurity { get; set; }
    [LoadColumn(9)] public string OnlineBackup { get; set; }
    [LoadColumn(10)] public string DeviceProtection { get; set; }
    [LoadColumn(11)] public string TechSupport { get; set; }
    [LoadColumn(12)] public string StreamingTV { get; set; }
    [LoadColumn(13)] public string StreamingMovies { get; set; }
    [LoadColumn(14)] public string Contract { get; set; }
    [LoadColumn(15)] public string PaperlessBilling { get; set; }
    [LoadColumn(16)] public string PaymentMethod { get; set; }
    [LoadColumn(17)] public float MonthlyCharges { get; set; }
    [LoadColumn(18)] public float TotalCharges { get; set; }
    [LoadColumn(19)] public bool Churn { get; set; }
}