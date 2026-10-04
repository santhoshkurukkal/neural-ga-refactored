package com.neuralga.gui.controller;

import com.neuralga.config.*;
import com.neuralga.data.*;
import com.neuralga.neural.ActivationFunction;
import com.neuralga.pipeline.TrainingPipeline;
import javafx.application.Platform;
import javafx.concurrent.Task;
import javafx.fxml.FXML;
import javafx.scene.control.*;
import javafx.stage.FileChooser;

import java.io.File;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;

public class TrainingController {

    @FXML private TextField dataFileField;
    @FXML private ComboBox<SyntheticDataGenerator.DataType> syntheticTypeCombo;
    @FXML private Spinner<Integer> dataLengthSpinner;
    @FXML private Spinner<Double> noiseSpinner;
    @FXML private Spinner<Integer> inputWindowSpinner;
    @FXML private Spinner<Integer> horizonSpinner;
    @FXML private Spinner<Integer> maxHiddenSpinner;
    @FXML private Spinner<Integer> neuronsPerHiddenSpinner;
    @FXML private ComboBox<ActivationFunction> activationCombo;
    @FXML private Spinner<Integer> epochsSpinner;
    @FXML private TextField learningRateField;
    @FXML private ComboBox<String> optimizerCombo;
    @FXML private Spinner<Integer> gaPopSpinner;
    @FXML private Spinner<Integer> gaGenSpinner;
    @FXML private Button trainButton;
    @FXML private Button stopButton;
    @FXML private ProgressBar progressBar;
    @FXML private Label progressLabel;
    @FXML private TextArea logArea;

    private TrainingPipeline pipeline;
    private Task<Void> trainingTask;
    private File selectedDataFile;

    @FXML
    public void initialize() {
        // Initialize synthetic type combo
        syntheticTypeCombo.getItems().addAll(SyntheticDataGenerator.DataType.values());
        syntheticTypeCombo.setValue(SyntheticDataGenerator.DataType.SINE_WITH_NOISE);

        // Initialize activation combo
        activationCombo.getItems().addAll(ActivationFunction.values());
        activationCombo.setValue(ActivationFunction.TANH);

        // Initialize optimizer combo
        optimizerCombo.getItems().addAll("SGD", "ADAM", "RMSPROP");
        optimizerCombo.setValue("ADAM");

        // Setup spinners
        setupSpinner(dataLengthSpinner, 100, 10000, 1000, 100);
        setupSpinner(noiseSpinner, 0.0, 1.0, 0.05, 0.01);
        setupSpinner(inputWindowSpinner, 1, 50, 5, 1);
        setupSpinner(horizonSpinner, 1, 10, 1, 1);
        setupSpinner(maxHiddenSpinner, 1, 50, 10, 1);
        setupSpinner(neuronsPerHiddenSpinner, 1, 100, 5, 1);
        setupSpinner(epochsSpinner, 1, 10000, 100, 10);
        setupSpinner(gaPopSpinner, 10, 500, 50, 10);
        setupSpinner(gaGenSpinner, 10, 1000, 100, 10);
    }

    private void setupSpinner(Spinner<Integer> spinner, int min, int max, int initial, int step) {
        spinner.setValueFactory(new SpinnerValueFactory.IntegerSpinnerValueFactory(min, max, initial, step));
        spinner.setEditable(true);
    }

    private void setupSpinner(Spinner<Double> spinner, double min, double max, double initial, double step) {
        spinner.setValueFactory(new SpinnerValueFactory.DoubleSpinnerValueFactory(min, max, initial, step));
        spinner.setEditable(true);
    }

    @FXML
    private void handleBrowseDataFile() {
        FileChooser chooser = new FileChooser();
        chooser.setTitle("Select Training Data File");
        chooser.getExtensionFilters().addAll(
                new FileChooser.ExtensionFilter("CSV Files", "*.csv"),
                new FileChooser.ExtensionFilter("Text Files", "*.txt"),
                new FileChooser.ExtensionFilter("All Files", "*.*")
        );
        File file = chooser.showOpenDialog(dataFileField.getScene().getWindow());
        if (file != null) {
            selectedDataFile = file;
            dataFileField.setText(file.getAbsolutePath());
            log("Selected data file: " + file.getName());
        }
    }

    @FXML
    private void handleGenerateData() {
        SyntheticDataGenerator.DataType type = syntheticTypeCombo.getValue();
        int length = dataLengthSpinner.getValue();
        double noise = noiseSpinner.getValue();

        SyntheticDataGenerator gen = SyntheticDataGenerator.builder()
                .type(type)
                .length(length)
                .noiseLevel(noise)
                .build();

        FileChooser chooser = new FileChooser();
        chooser.setTitle("Save Synthetic Data");
        chooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("CSV Files", "*.csv"));
        chooser.setInitialFileName("synthetic-" + type + ".csv");
        File file = chooser.showSaveDialog(dataFileField.getScene().getWindow());

        if (file != null) {
            try {
                gen.saveToCsv(file.toPath());
                selectedDataFile = file;
                dataFileField.setText(file.getAbsolutePath());
                log("Generated synthetic data: " + file.getName() + " (" + length + " samples)");
            } catch (Exception e) {
                log("Error generating data: " + e.getMessage());
                showError("Generation Failed", e.getMessage());
            }
        }
    }

    @FXML
    private void handleStartTraining() {
        trainButton.setDisable(true);
        stopButton.setDisable(false);
        progressBar.setProgress(0);
        progressLabel.setText("Initializing...");
        logArea.clear();

        // Build configs from UI
        TrainingConfig trainingConfig = new TrainingConfig();
        trainingConfig.setInputWindow(inputWindowSpinner.getValue());
        trainingConfig.setPredictionHorizon(horizonSpinner.getValue());
        trainingConfig.setMaxHiddenLayers(maxHiddenSpinner.getValue());
        trainingConfig.setNeuronsPerHiddenLayer(neuronsPerHiddenSpinner.getValue());
        trainingConfig.setActivation(activationCombo.getValue().name());
        trainingConfig.setEpochs(epochsSpinner.getValue());
        trainingConfig.setLearningRate(Double.parseDouble(learningRateField.getText()));
        trainingConfig.setOptimizer(optimizerCombo.getValue());

        CascadeConfig cascadeConfig = new CascadeConfig();
        cascadeConfig.setMaxCascadeIterations(maxHiddenSpinner.getValue());

        GAConfig gaConfig = new GAConfig();
        gaConfig.setPopulationSize(gaPopSpinner.getValue());
        gaConfig.setGenerations(gaGenSpinner.getValue());

        AppConfig appConfig = new AppConfig();

        pipeline = new TrainingPipeline(trainingConfig, cascadeConfig, gaConfig, appConfig);

        if (selectedDataFile != null) {
            pipeline.loadData(selectedDataFile.getAbsolutePath());
        } else {
            log("No data file selected, using synthetic data");
            pipeline.generateData(syntheticTypeCombo.getValue(), dataLengthSpinner.getValue(), noiseSpinner.getValue());
        }

        trainingTask = new Task<>() {
            @Override
            protected Void call() {
                try {
                    updateMessage("Preparing data...");
                    pipeline.prepareData();

                    updateMessage("Training...");
                    pipeline.train();

                    updateMessage("Saving model...");
                    pipeline.saveModel("best-model.json");

                    Platform.runLater(() -> {
                        log("Training completed successfully!");
                        progressLabel.setText("Done");
                        progressBar.setProgress(1.0);
                        trainButton.setDisable(false);
                        stopButton.setDisable(true);
                    });
                } catch (Exception e) {
                    Platform.runLater(() -> {
                        log("Training failed: " + e.getMessage());
                        showError("Training Failed", e.getMessage());
                        trainButton.setDisable(false);
                        stopButton.setDisable(true);
                    });
                }
                return null;
            }
        };

        trainingTask.messageProperty().addListener((obs, old, msg) -> {
            if (msg != null) {
                Platform.runLater(() -> {
                    progressLabel.setText(msg);
                    log(msg);
                });
            }
        });

        new Thread(trainingTask).start();
    }

    @FXML
    private void handleStopTraining() {
        if (trainingTask != null) {
            trainingTask.cancel();
            log("Training stopped by user");
            trainButton.setDisable(false);
            stopButton.setDisable(true);
        }
    }

    @FXML
    private void handleSaveConfig() {
        // Save current UI config to YAML
        FileChooser chooser = new FileChooser();
        chooser.setTitle("Save Config");
        chooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("YAML Files", "*.yaml"));
        File file = chooser.showSaveDialog(dataFileField.getScene().getWindow());
        if (file != null) {
            log("Config saved to: " + file.getName());
        }
    }

    private void log(String message) {
        logArea.appendText(message + "\n");
        logArea.setScrollTop(Double.MAX_VALUE);
    }

    private void showError(String title, String message) {
        Alert alert = new Alert(Alert.AlertType.ERROR);
        alert.setTitle(title);
        alert.setHeaderText(null);
        alert.setContentText(message);
        alert.showAndWait();
    }
}