package com.neuralga.gui.controller;

import javafx.fxml.FXML;
import javafx.scene.control.*;
import javafx.stage.Stage;

public class MainController {

    @FXML private TabPane tabPane;
    @FXML private Tab trainingTab;
    @FXML private Tab predictionTab;
    @FXML private Label statusLabel;

    @FXML
    public void initialize() {
        // Initialize tabs
    }

    @FXML
    private void handleNewProject() {
        statusLabel.setText("New project created");
    }

    @FXML
    private void handleOpenModel() {
        // Handled by prediction controller
    }

    @FXML
    private void handleSaveModel() {
        // Handled by training controller
    }

    @FXML
    private void handleExit() {
        Stage stage = (Stage) tabPane.getScene().getWindow();
        stage.close();
    }

    @FXML
    private void showTrainingView() {
        tabPane.getSelectionModel().select(trainingTab);
    }

    @FXML
    private void showPredictionView() {
        tabPane.getSelectionModel().select(predictionTab);
    }

    @FXML
    private void showModelInfo() {
        // Show model info dialog
    }

    @FXML
    private void handleAbout() {
        Alert alert = new Alert(Alert.AlertType.INFORMATION);
        alert.setTitle("About Neural-GA");
        alert.setHeaderText("Neural-GA: Cascade Correlation + GA Time-Series Predictor");
        alert.setContentText("Version 1.0.0\nRefactored from B.Tech project (2011)\n\nOptimization of Resources in Distributed System\nusing State Prediction Methods");
        alert.showAndWait();
    }

    public void setStatus(String status) {
        statusLabel.setText(status);
    }
}