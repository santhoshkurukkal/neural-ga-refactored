package com.neuralga.ga;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Random;

public class Population {

    private final List<Chromosome> individuals;
    private final Random random;

    public Population(int populationSize, int chromosomeLength, long seed) {
        this.individuals = new ArrayList<>(populationSize);
        this.random = new Random(seed);
        for (int i = 0; i < populationSize; i++) {
            individuals.add(new Chromosome(chromosomeLength, random));
        }
    }

    public Population(List<Chromosome> individuals, long seed) {
        this.individuals = new ArrayList<>(individuals);
        this.random = new Random(seed);
    }

    public int size() { return individuals.size(); }
    public Chromosome get(int index) { return individuals.get(index); }
    public List<Chromosome> getAll() { return List.copyOf(individuals); }

    public void add(Chromosome chromosome) {
        individuals.add(chromosome);
    }

    public void sortByFitness() {
        individuals.sort(Comparator.comparingDouble(Chromosome::getFitness).reversed());
    }

    public Chromosome getBest() {
        if (individuals.isEmpty()) return null;
        return individuals.stream()
                .max(Comparator.comparingDouble(Chromosome::getFitness))
                .orElse(null);
    }

    public Chromosome getWorst() {
        if (individuals.isEmpty()) return null;
        return individuals.stream()
                .min(Comparator.comparingDouble(Chromosome::getFitness))
                .orElse(null);
    }

    public double getAverageFitness() {
        if (individuals.isEmpty()) return 0.0;
        return individuals.stream().mapToDouble(Chromosome::getFitness).average().orElse(0.0);
    }

    public double getMaxFitness() {
        if (individuals.isEmpty()) return 0.0;
        return individuals.stream().mapToDouble(Chromosome::getFitness).max().orElse(0.0);
    }

    public double getMinFitness() {
        if (individuals.isEmpty()) return 0.0;
        return individuals.stream().mapToDouble(Chromosome::getFitness).min().orElse(0.0);
    }

    public double getFitnessStdDev() {
        if (individuals.size() < 2) return 0.0;
        double avg = getAverageFitness();
        double sumSq = individuals.stream()
                .mapToDouble(c -> Math.pow(c.getFitness() - avg, 2))
                .sum();
        return Math.sqrt(sumSq / individuals.size());
    }

    public void replaceWorst(Chromosome chromosome) {
        if (individuals.isEmpty()) return;
        int worstIndex = 0;
        double worstFitness = individuals.get(0).getFitness();
        for (int i = 1; i < individuals.size(); i++) {
            if (individuals.get(i).getFitness() < worstFitness) {
                worstFitness = individuals.get(i).getFitness();
                worstIndex = i;
            }
        }
        individuals.set(worstIndex, chromosome);
    }

    public Chromosome tournamentSelection(int tournamentSize) {
        if (individuals.isEmpty()) return null;
        Chromosome best = null;
        for (int i = 0; i < tournamentSize; i++) {
            Chromosome candidate = individuals.get(random.nextInt(individuals.size()));
            if (best == null || candidate.getFitness() > best.getFitness()) {
                best = candidate;
            }
        }
        return best;
    }

    public Chromosome rouletteWheelSelection() {
        if (individuals.isEmpty()) return null;

        double totalFitness = individuals.stream().mapToDouble(Chromosome::getFitness).sum();
        if (totalFitness <= 0) {
            return individuals.get(random.nextInt(individuals.size()));
        }

        double pick = random.nextDouble() * totalFitness;
        double cumulative = 0.0;
        for (Chromosome c : individuals) {
            cumulative += c.getFitness();
            if (cumulative >= pick) {
                return c;
            }
        }
        return individuals.get(individuals.size() - 1);
    }

    public void clear() {
        individuals.clear();
    }

    @Override
    public String toString() {
        return String.format("Population{size=%d, avgFitness=%.6f, maxFitness=%.6f}",
                size(), getAverageFitness(), getMaxFitness());
    }
}