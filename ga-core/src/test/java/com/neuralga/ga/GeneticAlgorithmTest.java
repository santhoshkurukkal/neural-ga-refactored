package com.neuralga.ga;

import com.neuralga.config.GAConfig;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.*;

class GeneticAlgorithmTest {

    @Test
    void shouldMinimizeSphereFunction() {
        // Sphere function: f(x) = sum(x_i^2), minimum at 0
        GAConfig config = new GAConfig();
        config.setPopulationSize(50);
        config.setGenerations(100);
        config.setElitismCount(2);
        config.setTournamentSize(3);
        config.setCrossoverRate(0.8);
        config.setMutationRate(0.1);
        config.setCrossoverOperator("SBX");
        config.setMutationOperator("GAUSSIAN");
        config.getTermination().setMaxGenerations(100);

        int dimensions = 5;

        FitnessFunction sphere = chromosome -> {
            double sum = 0;
            for (double g : chromosome.getGenes()) {
                sum += g * g;
            }
            return -sum; // Negative because GA maximizes
        };

        GeneticAlgorithm ga = new GeneticAlgorithm(config, sphere);
        ga.setChromosomeLength(dimensions);

        Chromosome best = ga.run();

        assertThat(best).isNotNull();
        assertThat(best.getLength()).isEqualTo(dimensions);
        // Fitness should be close to 0 (negative because we negated)
        assertThat(best.getFitness()).isGreaterThan(-1.0);
    }

    @Test
    void shouldRunWithListeners() {
        GAConfig config = new GAConfig();
        config.setPopulationSize(10);
        config.setGenerations(5);
        config.getTermination().setMaxGenerations(5);

        FitnessFunction fn = c -> 1.0;

        GeneticAlgorithm ga = new GeneticAlgorithm(config, fn);
        ga.setChromosomeLength(3);
        ga.addListener(new GeneticAlgorithm.GAListener() {
            int startCount = 0;
            int endCount = 0;

            @Override
            public void onGenerationStart(int generation, Population population) {
                startCount++;
            }

            @Override
            public void onGenerationEnd(int generation, Population population, Chromosome bestEver) {
                endCount++;
            }
        });

        Chromosome best = ga.run();
        assertThat(best).isNotNull();
    }
}