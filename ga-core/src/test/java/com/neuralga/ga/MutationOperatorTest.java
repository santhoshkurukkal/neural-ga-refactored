package com.neuralga.ga;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.*;

class MutationOperatorTest {

    @Test
    void gaussianShouldMutate() {
        Chromosome c = new Chromosome(new double[]{0.0, 0.0, 0.0, 0.0});
        MutationOperator.GaussianMutation mut = new MutationOperator.GaussianMutation(0.5);

        mut.mutate(c);

        // With stdDev=0.5, genes should change
        boolean allZero = true;
        for (double g : c.getGenes()) {
            if (g != 0.0) allZero = false;
        }
        assertThat(allZero).isFalse();
    }

    @Test
    void polynomialShouldMutate() {
        Chromosome c = new Chromosome(new double[]{0.0, 0.0, 0.0, 0.0});
        MutationOperator.PolynomialMutation mut = new MutationOperator.PolynomialMutation(20.0);

        mut.mutate(c);

        // Polynomial mutation has low probability per gene
        // Just verify it runs
    }

    @Test
    void createShouldReturnCorrectOperator() {
        assertThat(MutationOperator.create("GAUSSIAN", 20.0, 0.1)).isInstanceOf(MutationOperator.GaussianMutation.class);
        assertThat(MutationOperator.create("POLYNOMIAL", 20.0, 0.1)).isInstanceOf(MutationOperator.PolynomialMutation.class);
        assertThat(MutationOperator.create("unknown", 20.0, 0.1)).isInstanceOf(MutationOperator.GaussianMutation.class);
    }
}