package com.neuralga.ga;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.Arrays;
import java.util.Random;

public class Chromosome {

    private final double[] genes;
    private double fitness;
    private boolean evaluated;

    @JsonCreator
    public Chromosome(@JsonProperty("genes") double[] genes) {
        this.genes = genes != null ? genes.clone() : new double[0];
        this.fitness = 0.0;
        this.evaluated = false;
    }

    public Chromosome(int length, Random random) {
        this.genes = new double[length];
        this.fitness = 0.0;
        this.evaluated = false;
        // Initialize with small random values
        for (int i = 0; i < length; i++) {
            genes[i] = random.nextGaussian() * 0.1;
        }
    }

    public Chromosome(double[] genes, double fitness) {
        this.genes = genes != null ? genes.clone() : new double[0];
        this.fitness = fitness;
        this.evaluated = true;
    }

    public int getLength() { return genes.length; }
    public double[] getGenes() { return genes.clone(); }
    public double getGene(int index) { return genes[index]; }
    public void setGene(int index, double value) { genes[index] = value; }

    public double getFitness() { return fitness; }
    public void setFitness(double fitness) {
        this.fitness = fitness;
        this.evaluated = true;
    }

    public boolean isEvaluated() { return evaluated; }

    public Chromosome copy() {
        return new Chromosome(genes, fitness);
    }

    public void mutate(double mutationRate, double stdDev, Random random) {
        for (int i = 0; i < genes.length; i++) {
            if (random.nextDouble() < mutationRate) {
                genes[i] += random.nextGaussian() * stdDev;
            }
        }
        evaluated = false;
    }

    @Override
    public String toString() {
        return String.format("Chromosome{length=%d, fitness=%.6f, evaluated=%b}",
                genes.length, fitness, evaluated);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Chromosome that = (Chromosome) o;
        return Arrays.equals(genes, that.genes);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(genes);
    }
}