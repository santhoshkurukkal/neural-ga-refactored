package com.neuralga.cli.command;

import com.neuralga.data.CsvTimeseriesReader;
import com.neuralga.data.SlidingWindowDataset;
import com.neuralga.service.ModelLoader;
import com.neuralga.service.PredictionService;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.Callable;

@Command(name = "evaluate", description = "Evaluate a trained model on test data")
public class EvaluateCommand implements Callable<Integer> {

    @Option(names = {"-m", "--model"}, description = "Path to model file", required = true)
    String modelFile;

    @Option(names = {"-d", "--data"}, description = "Path to test data file", required = true)
    String dataFile;

    @Override
    public Integer call() {
        try {
            Path modelPath = Paths.get(modelFile);
            ModelLoader.Model model = ModelLoader.load(modelPath);
            PredictionService service = new PredictionService(model);

            CsvTimeseriesReader reader = new CsvTimeseriesReader();
            var data = reader.loadFromPath(Paths.get(dataFile));

            SlidingWindowDataset dataset = SlidingWindowDataset.builder()
                    .inputWindow(model.getInputWindow())
                    .predictionHorizon(model.getPredictionHorizon())
                    .build();
            dataset.build(data);

            PredictionService.BatchPredictor predictor = new PredictionService.BatchPredictor(service);
            var predictions = predictor.predictToList(dataset);

            // Calculate metrics
            double mse = 0, mae = 0;
            for (int i = 0; i < predictions.size(); i++) {
                double target = dataset.getTarget(i)[0];
                double pred = predictions.get(i)[0];
                double diff = pred - target;
                mse += diff * diff;
                mae += Math.abs(diff);
            }
            mse /= predictions.size();
            mae /= predictions.size();

            System.out.printf("Test MSE: %.6f%n", mse);
            System.out.printf("Test MAE: %.6f%n", mae);

            return 0;
        } catch (Exception e) {
            System.err.println("Evaluation failed: " + e.getMessage());
            e.printStackTrace();
            return 1;
        }
    }
}