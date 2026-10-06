using Microsoft.AspNetCore.Mvc;
using TelcoChurnWebApp.Models;

namespace TelcoChurnWebApp.Controllers
{
    public class ChurnController : Controller
    {
        private readonly ChurnPredictionService _predictionService;

        public ChurnController(ChurnPredictionService predictionService)
        {
            _predictionService = predictionService;
        }

        [HttpGet]
        public IActionResult Index()
        {
            return View(new ModelInput());
        }

        [HttpPost]
        public IActionResult Index(ModelInput input)
        {
            var result = _predictionService.Predict(input);
            ViewBag.Prediction = result.Prediction;
            ViewBag.Probability = result.Probability;
            return View(input);
        }
    }
}