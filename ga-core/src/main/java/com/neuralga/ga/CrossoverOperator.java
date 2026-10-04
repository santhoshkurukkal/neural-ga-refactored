package com.neuralga.ga;

import java.util.Random;

public interface CrossoverOperator {

    Chromosome[] crossover(Chromosome parent1, Chromosome parent2);

    static CrossoverOperator create(String name, double distributionIndex) {
        if (name == null) return new SBXCrossover(15.0);
        return switch (name.toUpperCase()) {
            case "SBX" -> new SBXCrossover(distributionIndex);
            case "UNIFORM" -> new UniformCrossover();
            case "SINGLE_POINT" -> new SinglePointCrossover();
            default -> new SBXCrossover(distributionIndex);
        };
    }

    class SBXCrossover implements CrossoverOperator {
        private final double distributionIndex;

        public SBXCrossover(double distributionIndex) {
            this.distributionIndex = distributionIndex;
        }

        @Override
        public Chromosome[] crossover(Chromosome parent1, Chromosome parent2) {
            int length = parent1.getLength();
            double[] child1Genes = new double[length];
            double[] child2Genes = new double[length];
            Random random = new Random();

            for (int i = 0; i < length; i++) {
                double x1 = parent1.getGene(i);
                double x2 = parent2.getGene(i);

                if (random.nextDouble() <= 0.5) {
                    double beta = calculateBeta(random);
                    child1Genes[i] = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2);
                    child2Genes[i] = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2);
                } else {
                    child1Genes[i] = x1;
                    child2Genes[i] = x2;
                }
            }

            return new Chromosome[]{
                    new Chromosome(child1Genes),
                    new Chromosome(child2Genes)
            };
        }

        private double calculateBeta(Random random) {
            double u = random.nextDouble();
            if (u <= 0.5) {
                return Math.pow(2 * u, 1.0 / (distributionIndex + 1));
            } else {
                return Math.pow(1.0 / (2 * (1 - u)), 1.0 / (distributionIndex + 1));
            }
        }
    }

    class UniformCrossover implements CrossoverOperator {
        @Override
        public Chromosome[] crossover(Chromosome parent1, Chromosome parent2) {
            int length = parent1.getLength();
            double[] child1Genes = new double[length];
            double[] child2Genes = new double[length];
            Random random = new Random();

            for (int i = 0; i < length; i++) {
                if (random.nextBoolean()) {
                    child1Genes[i] = parent1.getGene(i);
                    child2Genes[i] = parent2.getGene(i);
                } else {
                    child1Genes[i] = parent2.getGene(i);
                    child2Genes[i] = parent1.getGene(i);
                }
            }

            return new Chromosome[]{
                    new Chromosome(child1Genes),
                    new Chromosome(child2Genes)
            };
        }
    }

    class SinglePointCrossover implements CrossoverOperator {
        @Override
        public Chromosome[] crossover(Chromosome parent1, Chromosome parent2) {
            int length = parent1.getLength();
            if (length < 2) {
                return new Chromosome[]{parent1.copy(), parent2.copy()};
            }

            Random random = new Random();
            int point = random.nextInt(length - 1) + 1;

            double[] child1Genes = new double[length];
            double[] child2Genes = new double[length];

            System.arraycopy(parent1.getGenes(), 0, child1Genes, 0, point);
            System.arraycopy(parent2.getGenes(), point, child1Genes, point, length - point);

            System.arraycopy(parent2.getGenes(), 0, child2Genes, 0, point);
            System.arraycopy(parent1.getGenes(), point, child2Genes, point, length - point);

            return new Chromosome[]{
                    new Chromosome(child1Genes),
                    new Chromosome(child2Genes)
            };
        }
    }
}