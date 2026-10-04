package com.neuralga.config;

import com.fasterxml.jackson.annotation.JsonProperty;

public class GAConfig {

    @JsonProperty("population_size")
    private int populationSize = 50;

    @JsonProperty("generations")
    private int generations = 100;

    @JsonProperty("elitism_count")
    private int elitismCount = 2;

    @JsonProperty("tournament_size")
    private int tournamentSize = 3;

    @JsonProperty("crossover_rate")
    private double crossoverRate = 0.8;

    @JsonProperty("mutation_rate")
    private double mutationRate = 0.1;

    @JsonProperty("crossover_operator")
    private String crossoverOperator = "SBX";

    @JsonProperty("mutation_operator")
    private String mutationOperator = "GAUSSIAN";

    @JsonProperty("fitness_metric")
    private String fitnessMetric = "VAL_MSE";

    @JsonProperty("sbx_distribution_index")
    private double sbxDistributionIndex = 15.0;

    @JsonProperty("polynomial_mutation_distribution_index")
    private double polynomialMutationDistributionIndex = 20.0;

    @JsonProperty("gaussian_mutation_std_dev")
    private double gaussianMutationStdDev = 0.1;

    @JsonProperty("termination")
    private TerminationConfig termination = new TerminationConfig();

    @JsonProperty("selection_strategy")
    private String selectionStrategy = "TOURNAMENT";

    @JsonProperty("random_seed")
    private long randomSeed = 42;

    @JsonProperty("parallel_ga_fitness")
    private boolean parallelGAFitness = true;

    public GAConfig() {}

    public int getPopulationSize() { return populationSize; }
    public void setPopulationSize(int populationSize) { this.populationSize = populationSize; }

    public int getGenerations() { return generations; }
    public void setGenerations(int generations) { this.generations = generations; }

    public int getElitismCount() { return elitismCount; }
    public void setElitismCount(int elitismCount) { this.elitismCount = elitismCount; }

    public int getTournamentSize() { return tournamentSize; }
    public void setTournamentSize(int tournamentSize) { this.tournamentSize = tournamentSize; }

    public double getCrossoverRate() { return crossoverRate; }
    public void setCrossoverRate(double crossoverRate) { this.crossoverRate = crossoverRate; }

    public double getMutationRate() { return mutationRate; }
    public void setMutationRate(double mutationRate) { this.mutationRate = mutationRate; }

    public String getCrossoverOperator() { return crossoverOperator; }
    public void setCrossoverOperator(String crossoverOperator) { this.crossoverOperator = crossoverOperator; }

    public String getMutationOperator() { return mutationOperator; }
    public void setMutationOperator(String mutationOperator) { this.mutationOperator = mutationOperator; }

    public String getFitnessMetric() { return fitnessMetric; }
    public void setFitnessMetric(String fitnessMetric) { this.fitnessMetric = fitnessMetric; }

    public String getSelectionStrategy() { return selectionStrategy; }
    public void setSelectionStrategy(String selectionStrategy) { this.selectionStrategy = selectionStrategy; }

    public long getRandomSeed() { return randomSeed; }
    public void setRandomSeed(long randomSeed) { this.randomSeed = randomSeed; }

    public boolean isParallelGAFitness() { return parallelGAFitness; }
    public void setParallelGAFitness(boolean parallelGAFitness) { this.parallelGAFitness = parallelGAFitness; }

    public double getSbxDistributionIndex() { return sbxDistributionIndex; }
    public void setSbxDistributionIndex(double sbxDistributionIndex) { this.sbxDistributionIndex = sbxDistributionIndex; }

    public double getPolynomialMutationDistributionIndex() { return polynomialMutationDistributionIndex; }
    public void setPolynomialMutationDistributionIndex(double polynomialMutationDistributionIndex) { this.polynomialMutationDistributionIndex = polynomialMutationDistributionIndex; }

    public double getGaussianMutationStdDev() { return gaussianMutationStdDev; }
    public void setGaussianMutationStdDev(double gaussianMutationStdDev) { this.gaussianMutationStdDev = gaussianMutationStdDev; }

    public TerminationConfig getTermination() { return termination; }
    public void setTermination(TerminationConfig termination) { this.termination = termination; }

    public static class TerminationConfig {
        @JsonProperty("max_generations")
        private int maxGenerations = 100;

        @JsonProperty("max_time_seconds")
        private long maxTimeSeconds = 300;

        @JsonProperty("convergence_threshold")
        private double convergenceThreshold = 1e-6;

        @JsonProperty("stagnation_generations")
        private int stagnationGenerations = 20;

        public TerminationConfig() {}

        public int getMaxGenerations() { return maxGenerations; }
        public void setMaxGenerations(int maxGenerations) { this.maxGenerations = maxGenerations; }

        public long getMaxTimeSeconds() { return maxTimeSeconds; }
        public void setMaxTimeSeconds(long maxTimeSeconds) { this.maxTimeSeconds = maxTimeSeconds; }

        public double getConvergenceThreshold() { return convergenceThreshold; }
        public void setConvergenceThreshold(double convergenceThreshold) { this.convergenceThreshold = convergenceThreshold; }

        public int getStagnationGenerations() { return stagnationGenerations; }
        public void setStagnationGenerations(int stagnationGenerations) { this.stagnationGenerations = stagnationGenerations; }
    }
}