package com.neuralga.data;

import com.neuralga.config.AppConfig;
import com.neuralga.config.TrainingConfig;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;

import static org.assertj.core.api.Assertions.*;

class DataPipelineTest {

    @TempDir
    Path tempDir;

    @Test
    void shouldLoadAndPrepareData() throws IOException {
        // Create test data file with enough data for validation/test splits
        Path dataDir = tempDir.resolve("data");
        dataDir.toFile().mkdirs();
        Path dataFile = dataDir.resolve("test.csv");
        // Need enough data for: train + val + test splits, each needing inputWindow + predictionHorizon samples
        // With inputWindow=3, horizon=1, need at least 4 per split. With 0.2+0.2=0.4 splits, need 4/0.4 = 10 per split
        // So need at least 40 total samples
        StringBuilder sb = new StringBuilder();
        for (int i = 1; i <= 100; i++) {
            sb.append(i).append(".0\n");
        }
        java.nio.file.Files.writeString(dataFile, sb.toString());

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
    void shouldSaveSyntheticData() throws IOException {
        TrainingConfig trainingConfig = new TrainingConfig();
        AppConfig appConfig = new AppConfig();
        Path dataDir = tempDir.resolve("data");
        dataDir.toFile().mkdirs();
        appConfig.setDataDir(dataDir.toString());

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