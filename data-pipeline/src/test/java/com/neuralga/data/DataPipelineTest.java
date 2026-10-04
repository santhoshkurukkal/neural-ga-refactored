package com.neuralga.data;

import com.neuralga.config.AppConfig;
import com.neuralga.config.TrainingConfig;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Path;
import java.util.List;

import static org.assertj.core.api.Assertions.*;

class DataPipelineTest {

    @TempDir
    Path tempDir;

    @Test
    void shouldLoadAndPrepareData() {
        // Create test data file
        Path dataDir = tempDir.resolve("data");
        dataDir.toFile().mkdirs();
        Path dataFile = dataDir.resolve("test.csv");
        java.nio.file.Files.writeString(dataFile, "1.0\n2.0\n3.0\n4.0\n5.0\n6.0\n7.0\n8.0\n9.0\n10.0\n");

        // Config
        TrainingConfig trainingConfig = new TrainingConfig();
        trainingConfig.setInputWindow(3);
        trainingConfig.setPredictionHorizon(1);
        trainingConfig.setValidationSplit(0.2);
        trainingConfig.setTestSplit(0.2);

        AppConfig appConfig = new AppConfig();
        appConfig.setDataDir(dataDir.toString());

        // Pipeline
        DataPipeline pipeline = new DataPipeline(trainingConfig, appConfig);
        DataPipeline.PreparedData prepared = pipeline
                .loadData("test.csv")
                .split()
                .normalize()
                .prepare();

        // Verify
        assertThat(prepared.getTrainDataset().size()).isGreaterThan(0);
        assertThat(prepared.getValDataset().size()).isGreaterThan(0);
        assertThat(prepared.getTestDataset().size()).isGreaterThan(0);

        // Check normalization
        DataNormalizer normalizer = prepared.getNormalizer();
        assertThat(normalizer.isFitted()).isTrue();
        assertThat(normalizer.getMethod()).isEqualTo(DataNormalizer.Method.Z_SCORE);
    }

    @Test
    void shouldGenerateAndPrepareSyntheticData() {
        TrainingConfig trainingConfig = new TrainingConfig();
        trainingConfig.setInputWindow(5);
        trainingConfig.setPredictionHorizon(1);
        trainingConfig.setValidationSplit(0.2);
        trainingConfig.setTestSplit(0.1);

        AppConfig appConfig = new AppConfig();
        appConfig.setDataDir(tempDir.resolve("data").toString());

        DataPipeline pipeline = new DataPipeline(trainingConfig, appConfig);
        DataPipeline.PreparedData prepared = pipeline
                .generateSyntheticData(SyntheticDataGenerator.DataType.SINE_WITH_NOISE, 200, 0.05)
                .split()
                .normalize()
                .prepare();

        assertThat(prepared.getTrainDataset().size()).isGreaterThan(0);
        assertThat(prepared.getValDataset().size()).isGreaterThan(0);
        assertThat(prepared.getTestDataset().size()).isGreaterThan(0);
    }

    @Test
    void shouldSaveSyntheticData() {
        TrainingConfig trainingConfig = new TrainingConfig();
        AppConfig appConfig = new AppConfig();
        appConfig.setDataDir(tempDir.resolve("data").toString());

        DataPipeline pipeline = new DataPipeline(trainingConfig, appConfig);
        pipeline.generateSyntheticData(SyntheticDataGenerator.DataType.SINE, 50, 0.0)
                .saveSyntheticData("saved.csv");

        Path savedFile = tempDir.resolve("data/saved.csv");
        assertThat(savedFile).exists();

        String content = java.nio.file.Files.readString(savedFile);
        String[] lines = content.trim().split("\n");
        assertThat(lines).hasSize(50);
    }

    @Test
    void shouldThrowWhenStepsOutOfOrder() {
        TrainingConfig trainingConfig = new TrainingConfig();
        AppConfig appConfig = new AppConfig();

        DataPipeline pipeline = new DataPipeline(trainingConfig, appConfig);

        assertThatThrownBy(() -> pipeline.split())
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("No data loaded");

        pipeline.generateSyntheticData(SyntheticDataGenerator.DataType.SINE, 100, 0.0);
        assertThatThrownBy(() -> pipeline.normalize())
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("not split");

        pipeline.split();
        assertThatThrownBy(() -> pipeline.prepare())
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("not prepared");
    }

    @Test
    void shouldProvideRawDataAccess() {
        TrainingConfig trainingConfig = new TrainingConfig();
        AppConfig appConfig = new AppConfig();
        appConfig.setDataDir(tempDir.resolve("data").toString());

        DataPipeline pipeline = new DataPipeline(trainingConfig, appConfig);
        pipeline.generateSyntheticData(SyntheticDataGenerator.DataType.SINE, 100, 0.0)
                .split();

        assertThat(pipeline.getRawData()).hasSize(100);
        assertThat(pipeline.getSplitResult().getTrain()).isNotEmpty();
        assertThat(pipeline.getSplitResult().getVal()).isNotEmpty();
        assertThat(pipeline.getSplitResult().getTest()).isNotEmpty();
    }
}