package com.neuralga.config;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;

import static org.assertj.core.api.Assertions.*;

class ConfigLoaderTest {

    @TempDir
    Path tempDir;

    @Test
    void shouldLoadTrainingConfigFromYaml() {
        Path configFile = tempDir.resolve("test-training.yaml");
        String yaml = """
            neural:
              input_window: 10
              prediction_horizon: 2
            training:
              epochs: 200
              learning_rate: 0.001
            """;
        writeFile(configFile, yaml);

        TrainingConfig config = ConfigLoader.loadTrainingConfig(configFile.toString());

        assertThat(config.getInputWindow()).isEqualTo(10);
        assertThat(config.getPredictionHorizon()).isEqualTo(2);
        assertThat(config.getEpochs()).isEqualTo(200);
        assertThat(config.getLearningRate()).isEqualTo(0.001);
        // Defaults should be used for unspecified values
        assertThat(config.getInitialOutputNeurons()).isEqualTo(5);
        assertThat(config.getActivation()).isEqualTo("TANH");
    }

    @Test
    void shouldLoadGAConfigFromYaml() {
        Path configFile = tempDir.resolve("test-ga.yaml");
        String yaml = """
            population_size: 100
            generations: 200
            crossover_operator: "UNIFORM"
            mutation_operator: "POLYNOMIAL"
            termination:
              max_generations: 200
            """;
        writeFile(configFile, yaml);

        GAConfig config = ConfigLoader.loadGAConfig(configFile.toString());

        assertThat(config.getPopulationSize()).isEqualTo(100);
        assertThat(config.getGenerations()).isEqualTo(200);
        assertThat(config.getCrossoverOperator()).isEqualTo("UNIFORM");
        assertThat(config.getMutationOperator()).isEqualTo("POLYNOMIAL");
        assertThat(config.getTermination().getMaxGenerations()).isEqualTo(200);
    }

    @Test
    void shouldLoadCascadeConfigFromYaml() {
        Path configFile = tempDir.resolve("test-cascade.yaml");
        String yaml = """
            error_threshold: 0.1
            max_cascade_iterations: 5
            """;
        writeFile(configFile, yaml);

        CascadeConfig config = ConfigLoader.loadCascadeConfig(configFile.toString());

        assertThat(config.getErrorThreshold()).isEqualTo(0.1);
        assertThat(config.getMaxCascadeIterations()).isEqualTo(5);
    }

    @Test
    void shouldLoadAppConfigFromYaml() {
        Path configFile = tempDir.resolve("test-app.yaml");
        String yaml = """
            data_dir: "custom/data"
            models_dir: "custom/models"
            random_seed: 12345
            """;
        writeFile(configFile, yaml);

        AppConfig config = ConfigLoader.loadAppConfig(configFile.toString());

        assertThat(config.getDataDir().toString()).isEqualTo("custom/data");
        assertThat(config.getModelsDir().toString()).isEqualTo("custom/models");
        assertThat(config.getRandomSeed()).isEqualTo(12345L);
    }

    @Test
    void shouldReturnDefaultsWhenConfigFileNotFound() {
        TrainingConfig config = ConfigLoader.loadTrainingConfig("nonexistent.yaml");

        assertThat(config).isNotNull();
        assertThat(config.getInputWindow()).isEqualTo(5);
        assertThat(config.getEpochs()).isEqualTo(100);
    }

    @Test
    void shouldSaveAndLoadConfig() {
        Path configFile = tempDir.resolve("save-test.yaml");
        TrainingConfig original = new TrainingConfig();
        original.setInputWindow(7);
        original.setEpochs(150);
        original.setLearningRate(0.005);

        ConfigLoader.saveConfig(original, configFile.toString());
        TrainingConfig loaded = ConfigLoader.loadTrainingConfig(configFile.toString());

        assertThat(loaded.getInputWindow()).isEqualTo(7);
        assertThat(loaded.getEpochs()).isEqualTo(150);
        assertThat(loaded.getLearningRate()).isEqualTo(0.005);
    }

    @Test
    void shouldLoadFromClasspathResource() {
        // This tests loading from src/main/resources
        TrainingConfig config = ConfigLoader.loadConfigFromResource("training.yaml", TrainingConfig.class);

        assertThat(config).isNotNull();
        assertThat(config.getInputWindow()).isEqualTo(5);
        assertThat(config.getEpochs()).isEqualTo(100);
    }

    private void writeFile(Path path, String content) {
        try {
            java.nio.file.Files.writeString(path, content);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }
}