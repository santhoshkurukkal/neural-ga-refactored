package com.neuralga.ga;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.*;

class PopulationTest {

    @Test
    void shouldCreatePopulation() {
        Population pop = new Population(10, 5, 42);

        assertThat(pop.size()).isEqualTo(10);
        assertThat(pop.get(0).getLength()).isEqualTo(5);
    }

    @Test
    void shouldSortByFitness() {
        Population pop = new Population(5, 3, 42);
        pop.get(0).setFitness(0.1);
        pop.get(1).setFitness(0.9);
        pop.get(2).setFitness(0.5);
        pop.get(3).setFitness(0.3);
        pop.get(4).setFitness(0.7);

        pop.sortByFitness();

        assertThat(pop.get(0).getFitness()).isEqualTo(0.9);
        assertThat(pop.get(1).getFitness()).isEqualTo(0.7);
        assertThat(pop.get(4).getFitness()).isEqualTo(0.1);
    }

    @Test
    void shouldGetBestAndWorst() {
        Population pop = new Population(5, 3, 42);
        pop.get(0).setFitness(0.1);
        pop.get(1).setFitness(0.9);
        pop.get(2).setFitness(0.5);

        assertThat(pop.getBest().getFitness()).isEqualTo(0.9);
        assertThat(pop.getWorst().getFitness()).isEqualTo(0.1);
    }

    @Test
    void shouldCalculateStats() {
        Population pop = new Population(4, 2, 42);
        pop.get(0).setFitness(1.0);
        pop.get(1).setFitness(2.0);
        pop.get(2).setFitness(3.0);
        pop.get(3).setFitness(4.0);

        assertThat(pop.getAverageFitness()).isEqualTo(2.5);
        assertThat(pop.getMaxFitness()).isEqualTo(4.0);
        assertThat(pop.getMinFitness()).isEqualTo(1.0);
        assertThat(pop.getFitnessStdDev()).isCloseTo(1.118, within(0.01));
    }

    @Test
    void shouldTournamentSelect() {
        Population pop = new Population(10, 3, 42);
        for (int i = 0; i < 10; i++) {
            pop.get(i).setFitness(i * 0.1);
        }

        Chromosome selected = pop.tournamentSelection(3);
        assertThat(selected).isNotNull();
        assertThat(selected.getFitness()).isGreaterThan(0.0);
    }

    @Test
    void shouldRouletteSelect() {
        Population pop = new Population(5, 3, 42);
        pop.get(0).setFitness(10.0);
        pop.get(1).setFitness(1.0);
        pop.get(2).setFitness(1.0);
        pop.get(3).setFitness(1.0);
        pop.get(4).setFitness(1.0);

        // Best should be selected more often
        int bestCount = 0;
        for (int i = 0; i < 100; i++) {
            if (pop.rouletteWheelSelection() == pop.get(0)) bestCount++;
        }
        assertThat(bestCount).isGreaterThan(50);
    }

    @Test
    void shouldReplaceWorst() {
        Population pop = new Population(3, 2, 42);
        pop.get(0).setFitness(0.1);
        pop.get(1).setFitness(0.5);
        pop.get(2).setFitness(0.9);

        Chromosome better = new Chromosome(new double[]{1,1});
        better.setFitness(1.0);

        pop.replaceWorst(better);

        assertThat(pop.getWorst().getFitness()).isEqualTo(0.5); // Previous 0.1 replaced
    }
}