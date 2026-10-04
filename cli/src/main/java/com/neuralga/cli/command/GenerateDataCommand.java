package com.neuralga.cli.command;

import com.neuralga.data.SyntheticDataGenerator;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.Callable;

@Command(name = "generate-data", description = "Generate synthetic training data")
public class GenerateDataCommand implements Callable<Integer> {

    @Option(names = {"-o", "--output"}, description = "Output file", required = true)
    String outputFile;

    @Option(names = {"--type"}, description = "Data type")
    SyntheticDataGenerator.DataType type = SyntheticDataGenerator.DataType.SINE_WITH_NOISE;

    @Option(names = {"--length"}, description = "Data length")
    int length = 1000;

    @Option(names = {"--noise"}, description = "Noise level")
    double noise = 0.05;

    @Option(names = {"--seed"}, description = "Random seed")
    long seed = System.currentTimeMillis();

    @Override
    public Integer call() {
        try {
            SyntheticDataGenerator gen = SyntheticDataGenerator.builder()
                    .type(type)
                    .length(length)
                    .noiseLevel(noise)
                    .seed(seed)
                    .build();

            gen.saveToCsv(Paths.get(outputFile));
            System.out.println("Generated " + length + " " + type + " samples to: " + outputFile);
            return 0;
        } catch (Exception e) {
            System.err.println("Generation failed: " + e.getMessage());
            e.printStackTrace();
            return 1;
        }
    }
}