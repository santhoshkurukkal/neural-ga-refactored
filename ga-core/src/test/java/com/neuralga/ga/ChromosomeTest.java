package com.neuralga.ga;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.*;

class ChromosomeTest {

    @Test
    void shouldCreateWithRandomGenes() {
        Chromosome c = new Chromosome(10, new java.util.Random(42));

        assertThat(c.getLength()).isEqualTo(10);
        assertThat(c.getGenes()).hasSize(10);
        assertThat(c.getFitness()).isEqualTo(0.0);
        assertThat(c.isEvaluated()).isFalse();
    }

    @Test
    void shouldCreateWithGivenGenes() {
        double[] genes = {1.0, 2.0, 3.0};
        Chromosome c = new Chromosome(genes);

        assertThat(c.getLength()).isEqualTo(3);
        assertThat(c.getGenes()).containsExactly(1.0, 2.0, 3.0);
    }

    @Test
    void shouldCopy() {
        Chromosome original = new Chromosome(new double[]{1.0, 2.0, 3.0});
        original.setFitness(0.95);

        Chromosome copy = original.copy();

        assertThat(copy.getGenes()).isEqualTo(original.getGenes());
        assertThat(copy.getFitness()).isEqualTo(0.95);
        assertThat(copy).isNotSameAs(original);
    }

    @Test
    void shouldMutate() {
        Chromosome c = new Chromosome(new double[]{0.0, 0.0, 0.0, 0.0});
        c.mutate(1.0, 0.1, new java.util.Random(42));

        // All genes should change with mutationRate=1.0
        assertThat(c.getGenes()).doesNotContain(0.0);
    }

    @Test
    void shouldSetGene() {
        Chromosome c = new Chromosome(5, new java.util.Random(42));
        c.setGene(2, 99.0);

        assertThat(c.getGene(2)).isEqualTo(99.0);
    }

    @Test
    void shouldEqualBasedOnGenes() {
        Chromosome c1 = new Chromosome(new double[]{1.0, 2.0, 3.0});
        Chromosome c2 = new Chromosome(new double[]{1.0, 2.0, 3.0});
        Chromosome c3 = new Chromosome(new double[]{1.0, 2.0, 4.0});

        assertThat(c1).isEqualTo(c2);
        assertThat(c1).isNotEqualTo(c3);
    }
}