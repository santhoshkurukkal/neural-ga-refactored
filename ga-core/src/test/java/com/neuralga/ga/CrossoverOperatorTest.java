package com.neuralga.ga;

import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.*;

class CrossoverOperatorTest {

    @Test
    void sbxShouldCreateChildren() {
        Chromosome p1 = new Chromosome(new double[]{1.0, 2.0, 3.0, 4.0});
        Chromosome p2 = new Chromosome(new double[]{5.0, 6.0, 7.0, 8.0});

        CrossoverOperator.SBXCrossover sbx = new CrossoverOperator.SBXCrossover(15.0);
        Chromosome[] children = sbx.crossover(p1, p2);

        assertThat(children).hasSize(2);
        assertThat(children[0].getLength()).isEqualTo(4);
        assertThat(children[1].getLength()).isEqualTo(4);

        // Children should be between parents
        for (int i = 0; i < 4; i++) {
            double c1 = children[0].getGene(i);
            double c2 = children[1].getGene(i);
            assertThat(c1).isBetween(1.0, 5.0);
            assertThat(c2).isBetween(1.0, 5.0);
        }
    }

    @Test
    void uniformShouldCreateChildren() {
        Chromosome p1 = new Chromosome(new double[]{1.0, 1.0, 1.0, 1.0});
        Chromosome p2 = new Chromosome(new double[]{2.0, 2.0, 2.0, 2.0});

        CrossoverOperator.UniformCrossover uniform = new CrossoverOperator.UniformCrossover();
        Chromosome[] children = uniform.crossover(p1, p2);

        assertThat(children).hasSize(2);
        for (Chromosome c : children) {
            for (int i = 0; i < 4; i++) {
                assertThat(c.getGene(i)).isIn(1.0, 2.0);
            }
        }
    }

    @Test
    void singlePointShouldCreateChildren() {
        Chromosome p1 = new Chromosome(new double[]{1.0, 1.0, 1.0, 1.0});
        Chromosome p2 = new Chromosome(new double[]{2.0, 2.0, 2.0, 2.0});

        CrossoverOperator.SinglePointCrossover sp = new CrossoverOperator.SinglePointCrossover();
        Chromosome[] children = sp.crossover(p1, p2);

        assertThat(children).hasSize(2);
        // Each child should have some genes from p1 and some from p2
        for (Chromosome c : children) {
            boolean has1 = false, has2 = false;
            for (int i = 0; i < 4; i++) {
                if (c.getGene(i) == 1.0) has1 = true;
                if (c.getGene(i) == 2.0) has2 = true;
            }
            assertThat(has1).isTrue();
            assertThat(has2).isTrue();
        }
    }

    @Test
    void createShouldReturnCorrectOperator() {
        assertThat(CrossoverOperator.create("SBX", 15.0)).isInstanceOf(CrossoverOperator.SBXCrossover.class);
        assertThat(CrossoverOperator.create("UNIFORM", 15.0)).isInstanceOf(CrossoverOperator.UniformCrossover.class);
        assertThat(CrossoverOperator.create("SINGLE_POINT", 15.0)).isInstanceOf(CrossoverOperator.SinglePointCrossover.class);
        assertThat(CrossoverOperator.create("unknown", 15.0)).isInstanceOf(CrossoverOperator.SBXCrossover.class);
    }
}