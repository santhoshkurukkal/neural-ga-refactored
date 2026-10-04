package com.neuralga.ga;

import java.util.Random;

public interface SelectionStrategy {

    Chromosome select(Population population);

    static SelectionStrategy create(String name, int tournamentSize) {
        if (name == null) return new TournamentSelection(3);
        return switch (name.toUpperCase()) {
            case "TOURNAMENT" -> new TournamentSelection(tournamentSize);
            case "ROULETTE" -> new RouletteWheelSelection();
            case "RANK" -> new RankSelection();
            default -> new TournamentSelection(tournamentSize);
        };
    }

    class TournamentSelection implements SelectionStrategy {
        private final int tournamentSize;

        public TournamentSelection(int tournamentSize) {
            this.tournamentSize = Math.max(2, tournamentSize);
        }

        @Override
        public Chromosome select(Population population) {
            return population.tournamentSelection(tournamentSize);
        }
    }

    class RouletteWheelSelection implements SelectionStrategy {
        @Override
        public Chromosome select(Population population) {
            return population.rouletteWheelSelection();
        }
    }

    class RankSelection implements SelectionStrategy {
        @Override
        public Chromosome select(Population population) {
            // Sort by fitness, then select based on rank
            population.sortByFitness();
            int n = population.size();
            double[] rankProbs = new double[n];
            double sum = n * (n + 1) / 2.0;
            for (int i = 0; i < n; i++) {
                rankProbs[i] = (n - i) / sum; // Best gets highest probability
            }

            double pick = population.getAll().get(0).getFitness() > 0 ? Math.random() : 0;
            double cum = 0;
            for (int i = 0; i < n; i++) {
                cum += rankProbs[i];
                if (pick <= cum) {
                    return population.get(i);
                }
            }
            return population.get(n - 1);
        }
    }
}