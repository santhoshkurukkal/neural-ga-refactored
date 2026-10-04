package com.neuralga.ga;

import java.util.Random;

public interface MutationOperator {

    void mutate(Chromosome chromosome);

    static MutationOperator create(String name, double distributionIndex, double stdDev) {
        if (name == null) return new GaussianMutation(stdDev);
        return switch (name.toUpperCase()) {
            case "GAUSSIAN" -> new GaussianMutation(stdDev);
            case "POLYNOMIAL" -> new PolynomialMutation(distributionIndex);
            default -> new GaussianMutation(stdDev);
        };
    }

    class GaussianMutation implements MutationOperator {
        private final double stdDev;
        private final Random random = new Random();

        public GaussianMutation(double stdDev) {
            this.stdDev = stdDev;
        }

        @Override
        public void mutate(Chromosome chromosome) {
            for (int i = 0; i < chromosome.getLength(); i++) {
                chromosome.setGene(i, chromosome.getGene(i) + random.nextGaussian() * stdDev);
            }
            chromosome.setFitness(0);
            chromosome.setEvaluated(false);
        }
    }

    class PolynomialMutation implements MutationOperator {
        private final double distributionIndex;
        private final Random random = new Random();

        public PolynomialMutation(double distributionIndex) {
            this.distributionIndex = distributionIndex;
        }

        @Override
        public void mutate(Chromosome chromosome) {
            for (int i = 0; i < chromosome.getLength(); i++) {
                if (random.nextDouble() < 1.0 / chromosome.getLength()) {
                    double u = random.nextDouble();
                    double delta;
                    if (u < 0.5) {
                        delta = Math.pow(2 * u, 1.0 / (distributionIndex + 1)) - 1;
                    } else {
                        delta = 1 - Math.pow(2 * (1 - u), 1.0 / (distributionIndex + 1));
                    }
                    chromosome.setGene(i, chromosome.getGene(i) + delta);
                }
            }
            chromosome.setFitness(0);
            chromosome.setEvaluated(false);
        }
    }
}