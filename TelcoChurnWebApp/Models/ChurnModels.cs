using Microsoft.ML.Data;

namespace TelcoChurnWebApp.Models
{
    public class ModelInput
    {
        [LoadColumn(0)] public string Gender { get; set; } = "";
        [LoadColumn(1)] public float SeniorCitizen { get; set; }
        [LoadColumn(2)] public string Partner { get; set; } = "";
        [LoadColumn(3)] public string Dependents { get; set; } = "";
        [LoadColumn(4)] public float Tenure { get; set; }
        [LoadColumn(5)] public string PhoneService { get; set; } = "";
        [LoadColumn(6)] public string MultipleLines { get; set; } = "";
        [LoadColumn(7)] public string InternetService { get; set; } = "";
        [LoadColumn(8)] public string OnlineSecurity { get; set; } = "";
        [LoadColumn(9)] public string OnlineBackup { get; set; } = "";
        [LoadColumn(10)] public string DeviceProtection { get; set; } = "";
        [LoadColumn(11)] public string TechSupport { get; set; } = "";
        [LoadColumn(12)] public string StreamingTV { get; set; } = "";
        [LoadColumn(13)] public string StreamingMovies { get; set; } = "";
        [LoadColumn(14)] public string Contract { get; set; } = "";
        [LoadColumn(15)] public string PaperlessBilling { get; set; } = "";
        [LoadColumn(16)] public string PaymentMethod { get; set; } = "";
        [LoadColumn(17)] public float MonthlyCharges { get; set; }
        [LoadColumn(18)] public float TotalCharges { get; set; }
        [LoadColumn(19)] public bool Churn { get; set; }
    }

    public class ModelOutput
    {
        [ColumnName("PredictedLabel")]
        public bool Prediction { get; set; }

        public float Probability { get; set; }
        public float Score { get; set; }
    }
}