package com.neuralga.ga;

/**
 * Functional interface for evaluating chromosome fitness.
 * Higher fitness values are better.
 */
@FunctionalInterface
public interface FitnessFunction {

    /**
     * Evaluate the fitness of a chromosome.
     * Higher fitness is better.
     * @param chromosome the chromosome to evaluate
     * @return fitness value (higher is better)
     */
    double evaluate(Chromosome chromosome);
}