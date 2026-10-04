package com.neuralga.gui.controller;

import com.neuralga.config.AppConfig;
import com.neuralga.data.CsvTimeseriesReader;
import com.neuralga.data.SlidingWindowDataset;
import com.neuralga.service.ModelLoader;
import com.neuralga.service.PredictionService;
import javafx.application.Platform;
import javafx.concurrent.Task;
import javafx.fxml.FXML;
import javafx.scene.control.*;
import javafx.stage.FileChooser;

import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;

public class PredictionController {

    @FXML private TextField modelFileField;
    @FXML private TextField predictDataFileField;
    @FXML private TextField predictOutputField;
    @FXML private TextField interactiveInputField;
    @FXML private Spinner<Integer> predictStepsSpinner;
    @FXML private Button predictBatchButton;
    @FXML private Button predictInteractiveButton;
    @FXML private TextArea resultArea;

    private ModelLoader.Model loadedModel;
    private PredictionService predictionService;
    private File selectedDataFile;
    private File selectedOutputFile;

    @FXML
    public void initialize() {
        predictStepsSpinner.setValueFactory(new javafx.scene.control.SpinnerValueFactory.IntegerSpinnerValueFactory(1, 100, 1));
    }

    @FXML
    private void handleLoadModel() {
        FileChooser chooser = new FileChooser();
        chooser.setTitle("Load Model");
        chooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("JSON Files", "*.json"));
        File file = chooser.showOpenDialog(modelFileField.getScene().getWindow());

        if (file != null) {
            try {
                loadedModel = ModelLoader.load(file.toPath());
                predictionService = new PredictionService(loadedModel);
                modelFileField.setText(file.getAbsolutePath());
                appendResult("Model loaded: " + file.getName());
                appendResult("Input window: " + loadedModel.getInputWindow());
                appendResult("Prediction horizon: " + loadedModel.getPredictionHorizon());
                appendResult("Hidden layers: " + loadedModel.getNetwork().getHiddenLayerCount());
            } catch (Exception e) {
                appendResult("Error loading model: " + e.getMessage());
                showError("Load Failed", e.getMessage());
            }
        }
    }

    @FXML
    private void handleBrowsePredictData() {
        FileChooser chooser = new FileChooser();
        chooser.setTitle("Select Input Data File");
        chooser.getExtensionFilters().addAll(
                new FileChooser.ExtensionFilter("CSV Files", "*.csv"),
                new FileChooser.ExtensionFilter("Text Files", "*.txt"),
                new FileChooser.ExtensionFilter("All Files", "*.*")
        );
        File file = chooser.showOpenDialog(predictDataFileField.getScene().getWindow());
        if (file != null) {
            selectedDataFile = file;
            predictDataFileField.setText(file.getAbsolutePath());
        }
    }

    @FXML
    private void handleBrowsePredictOutput() {
        FileChooser chooser = new FileChooser();
        chooser.setTitle("Save Predictions");
        chooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("CSV Files", "*.csv"));
        chooser.setInitialFileName("predictions.csv");
        File file = chooser.showSaveDialog(predictOutputField.getScene().getWindow());
        if (file != null) {
            selectedOutputFile = file;
            predictOutputField.setText(file.getAbsolutePath());
        }
    }

    @FXML
    private void handleBatchPredict() {
        if (predictionService == null) {
            showError("No Model", "Please load a model first");
            return;
        }
        if (selectedDataFile == null) {
            showError("No Data", "Please select an input data file");
            return;
        }
        if (selectedOutputFile == null) {
            showError("No Output", "Please specify an output file");
            return;
        }

        predictBatchButton.setDisable(true);
        appendResult("Starting batch prediction...");

        Task<Void> task = new Task<>() {
            @Override
            protected Void call() {
                try {
                    CsvTimeseriesReader reader = new CsvTimeseriesReader();
                    List<Double> data = reader.loadFromPath(selectedDataFile.toPath());

                    SlidingWindowDataset dataset = SlidingWindowDataset.builder()
                            .inputWindow(loadedModel.getInputWindow())
                            .predictionHorizon(loadedModel.getPredictionHorizon())
                            .build();
                    dataset.build(data);

                    PredictionService.BatchPredictor predictor = new PredictionService.BatchPredictor(predictionService);
                    List<double[]> predictions = predictor.predictToList(dataset);

                    // Write results
                    StringBuilder sb = new StringBuilder();
                    sb.append("input,prediction\n");
                    for (int i = 0; i < predictions.size(); i++) {
                        double[] input = dataset.getInput(i);
                        double[] pred = predictions.get(i);
                        double target = dataset.getTarget(i)[0];
                        sb.append(String.join(",", java.util.Arrays.stream(input).mapToObj(String::valueOf).toArray(String[]::new)))
                                .append(",").append(pred[0]).append(",").append(target).append("\n");
                    }
                    Files.writeString(selectedOutputFile.toPath(), sb.toString());

                    Platform.runLater(() -> {
                        appendResult("Batch prediction complete. Results saved to: " + selectedOutputFile.getName());
                        appendResult("Samples predicted: " + predictions.size());
                        predictBatchButton.setDisable(false);
                    });
                } catch (Exception e) {
                    Platform.runLater(() -> {
                        appendResult("Batch prediction failed: " + e.getMessage());
                        showError("Prediction Failed", e.getMessage());
                        predictBatchButton.setDisable(false);
                    });
                }
                return null;
            }
        };

        new Thread(task).start();
    }

    @FXML
    private void handleInteractivePredict() {
        if (predictionService == null) {
            showError("No Model", "Please load a model first");
            return;
        }

        String inputText = interactiveInputField.getText().trim();
        if (inputText.isEmpty()) {
            showError("No Input", "Please enter input values");
            return;
        }

        try {
            String[] parts = inputText.split("[,\\s]+");
            double[] input = new double[parts.length];
            for (int i = 0; i < parts.length; i++) {
                input[i] = Double.parseDouble(parts[i]);
            }

            if (input.length != loadedModel.getInputWindow()) {
                showError("Invalid Input", "Expected " + loadedModel.getInputWindow() + " values, got " + input.length);
                return;
            }

            int steps = predictStepsSpinner.getValue();
            List<Double> predictions = predictionService.predictSequence(input, steps);

            appendResult("Input: " + java.util.Arrays.toString(input));
            appendResult("Predictions (" + steps + " steps): " + predictions);
        } catch (NumberFormatException e) {
            showError("Invalid Input", "Please enter comma-separated numbers");
        } catch (Exception e) {
            showError("Prediction Failed", e.getMessage());
        }
    }

    private void appendResult(String message) {
        resultArea.appendText(message + "\n");
        resultArea.setScrollTop(Double.MAX_VALUE);
    }

    private void showError(String title, String message) {
        Alert alert = new Alert(Alert.AlertType.ERROR);
        alert.setTitle(title);
        alert.setHeaderText(null);
        alert.setContentText(message);
        alert.showAndWait();
    }
}