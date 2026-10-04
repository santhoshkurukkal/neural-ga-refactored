package com.neuralga.cli;

import com.neuralga.cli.command.*;
import picocli.CommandLine;
import picocli.CommandLine.Command;

import java.util.concurrent.Callable;

@Command(name = "neural-ga", mixinStandardHelpOptions = true, version = "1.0.0",
        description = "Cascade Correlation Neural Network with Genetic Algorithm for Time-Series Prediction",
        subcommands = {
                TrainCommand.class,
                PredictCommand.class,
                EvaluateCommand.class,
                GenerateDataCommand.class
        })
public class NeuralGAMain implements Callable<Integer> {

    public static void main(String[] args) {
        int exitCode = new CommandLine(new NeuralGAMain()).execute(args);
        System.exit(exitCode);
    }

    @Override
    public Integer call() {
        System.out.println("Neural-GA: Cascade Correlation + GA for Time-Series Prediction");
        System.out.println("Use --help to see available commands");
        return 0;
    }
}