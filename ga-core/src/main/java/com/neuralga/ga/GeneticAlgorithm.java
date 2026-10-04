package com.neuralga.ga;

import com.neuralga.config.GAConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ForkJoinPool;
import java.util.function.Function;

public class GeneticAlgorithm {

    private static final Logger log = LoggerFactory.getLogger(GeneticAlgorithm.class);

    private final GAConfig config;
    private final FitnessFunction fitnessFunction;
    private final SelectionStrategy selection;
    private final CrossoverOperator crossover;
    private final MutationOperator mutation;

    private Population population;
    private Chromosome bestEver;
    private int generation = 0;
    private long startTime;
    private int stagnationCounter = 0;
    private double lastBestFitness = Double.NEGATIVE_INFINITY;

    private final List<GAListener> listeners = new ArrayList<>();

    public GeneticAlgorithm(GAConfig config, FitnessFunction fitnessFunction) {
        this.config = config;
        this.fitnessFunction = fitnessFunction;
        this.selection = SelectionStrategy.create(config.getSelectionStrategy(), config.getTournamentSize());
        this.crossover = CrossoverOperator.create(config.getCrossoverOperator(), config.getSbxDistributionIndex());
        this.mutation = MutationOperator.create(config.getMutationOperator(), config.getPolynomialMutationDistributionIndex(), config.getGaussianMutationStdDev());
    }

    public void addListener(GAListener listener) {
        listeners.add(listener);
    }

    public Chromosome run() {
        startTime = System.currentTimeMillis();
        generation = 0;
        stagnationCounter = 0;
        lastBestFitness = Double.NEGATIVE_INFINITY;

        // Initialize population - chromosome length must be set via setChromosomeLength() before run()
        if (population == null) {
            throw new IllegalStateException("Chromosome length not set. Call setChromosomeLength() before run().");
        }

        log.info("Starting GA: popSize={}, generations={}, chromosomeLength={}",
                config.getPopulationSize(), config.getGenerations(), population.get(0).getLength());

        notifyGenerationStart();

        // Evaluate initial population
        evaluatePopulation(population);

        // Main loop
        while (!shouldTerminate()) {
            generation++;
            evolveGeneration();

            if (generation % 10 == 0 || generation == 1) {
                log.info("Generation {}: best={:.6f}, avg={:.6f}, stagnation={}",
                        generation, population.getMaxFitness(), population.getAverageFitness(), stagnationCounter);
            }

            notifyGenerationEnd();
        }

        log.info("GA finished after {} generations. Best fitness: {:.6f}", generation, bestEver.getFitness());
        return bestEver;
    }

    public void setChromosomeLength(int length) {
        if (population == null) {
            population = new Population(config.getPopulationSize(), length, config.getRandomSeed());
        }
    }

    private void evaluatePopulation(Population pop) {
        if (config.isParallelGAFitness()) {
            evaluateParallel(pop);
        } else {
            evaluateSequential(pop);
        }
        updateBestEver();
    }

    private void evaluateParallel(Population pop) {
        ForkJoinPool pool = ForkJoinPool.commonPool();
        pool.submit(() -> {
            pop.getAll().parallelStream().forEach(this::evaluateIndividual);
        }).join();
    }

    private void evaluateSequential(Population pop) {
        for (Chromosome c : pop.getAll()) {
            evaluateIndividual(c);
        }
    }

    private void evaluateIndividual(Chromosome chromosome) {
        if (!chromosome.isEvaluated()) {
            double fitness = fitnessFunction.evaluate(chromosome);
            chromosome.setFitness(fitness);
        }
    }

    private void evolveGeneration() {
        Population newPopulation = new Population(0, population.get(0).getLength(), System.nanoTime());

        // Elitism: copy best individuals
        population.sortByFitness();
        for (int i = 0; i < config.getElitismCount() && i < population.size(); i++) {
            newPopulation.add(population.get(i).copy());
        }

        // Generate offspring
        while (newPopulation.size() < config.getPopulationSize()) {
            Chromosome parent1 = selection.select(population);
            Chromosome parent2 = selection.select(population);

            if (parent1 == null || parent2 == null) continue;

            Chromosome[] children;
            if (Math.random() < config.getCrossoverRate()) {
                children = crossover.crossover(parent1, parent2);
            } else {
                children = new Chromosome[]{parent1.copy(), parent2.copy()};
            }

            for (Chromosome child : children) {
                if (Math.random() < config.getMutationRate()) {
                    mutation.mutate(child);
                }
                if (newPopulation.size() < config.getPopulationSize()) {
                    newPopulation.add(child);
                }
            }
        }

        population = newPopulation;
        evaluatePopulation(population);
    }

    private void updateBestEver() {
        Chromosome currentBest = population.getBest();
        if (currentBest != null && currentBest.getFitness() > (bestEver != null ? bestEver.getFitness() : Double.NEGATIVE_INFINITY)) {
            bestEver = currentBest.copy();
            stagnationCounter = 0;
            lastBestFitness = bestEver.getFitness();
        } else {
            stagnationCounter++;
        }
    }

    private boolean shouldTerminate() {
        GAConfig.TerminationConfig term = config.getTermination();

        if (generation >= term.getMaxGenerations()) return true;
        if (System.currentTimeMillis() - startTime > term.getMaxTimeSeconds() * 1000) return true;
        if (stagnationCounter >= term.getStagnationGenerations()) return true;
        if (bestEver != null && lastBestFitness > 0 &&
                Math.abs(bestEver.getFitness() - lastBestFitness) < term.getConvergenceThreshold()) {
            return true;
        }
        return false;
    }

    public Chromosome getBestSolution() { return bestEver; }
    public Population getPopulation() { return population; }
    public int getGeneration() { return generation; }
    public double getBestFitness() { return bestEver != null ? bestEver.getFitness() : 0.0; }

    private void notifyGenerationStart() {
        for (GAListener l : listeners) l.onGenerationStart(generation, population);
    }

    private void notifyGenerationEnd() {
        for (GAListener l : listeners) l.onGenerationEnd(generation, population, bestEver);
    }

    public interface GAListener {
        void onGenerationStart(int generation, Population population);
        void onGenerationEnd(int generation, Population population, Chromosome bestEver);
    }
}