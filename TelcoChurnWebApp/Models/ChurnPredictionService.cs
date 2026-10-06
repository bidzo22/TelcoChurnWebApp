using Microsoft.ML;

namespace TelcoChurnWebApp.Models
{
    public class ChurnPredictionService
    {
        private readonly MLContext _mlContext;
        private readonly ITransformer _model;

        public ChurnPredictionService()
        {
            _mlContext = new MLContext();
            string modelPath = Path.Combine(AppContext.BaseDirectory, "MLModel", "ChurnModel.zip");
            _model = _mlContext.Model.Load(modelPath, out _);
        }

        public ModelOutput Predict(ModelInput input)
        {
            var predictionEngine = _mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(_model);
            return predictionEngine.Predict(input);
        }
    }
}