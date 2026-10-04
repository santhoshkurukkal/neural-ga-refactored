package com.neuralga.config;

import com.fasterxml.jackson.annotation.JsonProperty;

public class CascadeConfig {

    @JsonProperty("error_threshold")
    private double errorThreshold = 0.5;

    @JsonProperty("max_cascade_iterations")
    private int maxCascadeIterations = 10;

    @JsonProperty("retrain_epochs_after_growth")
    private int retrainEpochsAfterGrowth = 50;

    @JsonProperty("fine_tune_learning_rate_factor")
    private double fineTuneLearningRateFactor = 0.1;

    @JsonProperty("min_improvement_for_growth")
    private double minImprovementForGrowth = 0.01;

    public CascadeConfig() {}

    public double getErrorThreshold() { return errorThreshold; }
    public void setErrorThreshold(double errorThreshold) { this.errorThreshold = errorThreshold; }

    public int getMaxCascadeIterations() { return maxCascadeIterations; }
    public void setMaxCascadeIterations(int maxCascadeIterations) { this.maxCascadeIterations = maxCascadeIterations; }

    public int getRetrainEpochsAfterGrowth() { return retrainEpochsAfterGrowth; }
    public void setRetrainEpochsAfterGrowth(int retrainEpochsAfterGrowth) { this.retrainEpochsAfterGrowth = retrainEpochsAfterGrowth; }

    public double getFineTuneLearningRateFactor() { return fineTuneLearningRateFactor; }
    public void setFineTuneLearningRateFactor(double fineTuneLearningRateFactor) { this.fineTuneLearningRateFactor = fineTuneLearningRateFactor; }

    public double getMinImprovementForGrowth() { return minImprovementForGrowth; }
    public void setMinImprovementForGrowth(double minImprovementForGrowth) { this.minImprovementForGrowth = minImprovementForGrowth; }
}