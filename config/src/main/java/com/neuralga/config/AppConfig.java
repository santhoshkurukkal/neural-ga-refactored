package com.neuralga.config;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.nio.file.Path;
import java.nio.file.Paths;

public class AppConfig {

    @JsonProperty("data_dir")
    private String dataDir = "data/sample";

    @JsonProperty("models_dir")
    private String modelsDir = "models";

    @JsonProperty("logs_dir")
    private String logsDir = "logs";

    @JsonProperty("default_data_file")
    private String defaultDataFile = "cpu.txt";

    @JsonProperty("synthetic_data_file")
    private String syntheticDataFile = "synthetic.csv";

    @JsonProperty("default_model_file")
    private String defaultModelFile = "best-model.json";

    @JsonProperty("random_seed")
    private long randomSeed = System.currentTimeMillis();

    @JsonProperty("parallel_ga_fitness")
    private boolean parallelGAFitness = true;

    @JsonProperty("log_level")
    private String logLevel = "INFO";

    public AppConfig() {}

    public Path getDataDir() { return Paths.get(dataDir); }
    public void setDataDir(String dataDir) { this.dataDir = dataDir; }

    public Path getModelsDir() { return Paths.get(modelsDir); }
    public void setModelsDir(String modelsDir) { this.modelsDir = modelsDir; }

    public Path getLogsDir() { return Paths.get(logsDir); }
    public void setLogsDir(String logsDir) { this.logsDir = logsDir; }

    public String getDefaultDataFile() { return defaultDataFile; }
    public void setDefaultDataFile(String defaultDataFile) { this.defaultDataFile = defaultDataFile; }

    public String getSyntheticDataFile() { return syntheticDataFile; }
    public void setSyntheticDataFile(String syntheticDataFile) { this.syntheticDataFile = syntheticDataFile; }

    public String getDefaultModelFile() { return defaultModelFile; }
    public void setDefaultModelFile(String defaultModelFile) { this.defaultModelFile = defaultModelFile; }

    public long getRandomSeed() { return randomSeed; }
    public void setRandomSeed(long randomSeed) { this.randomSeed = randomSeed; }

    public boolean isParallelGAFitness() { return parallelGAFitness; }
    public void setParallelGAFitness(boolean parallelGAFitness) { this.parallelGAFitness = parallelGAFitness; }

    public String getLogLevel() { return logLevel; }
    public void setLogLevel(String logLevel) { this.logLevel = logLevel; }
}