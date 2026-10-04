package com.neuralga.neural;

public class EarlyStopping {

    private final int patience;
    private final double minDelta;
    private final boolean restoreBestWeights;

    private int counter = 0;
    private double bestScore = Double.POSITIVE_INFINITY;
    private Network.NetworkParams bestWeights;
    private boolean stopped = false;

    public EarlyStopping(int patience, double minDelta) {
        this(patience, minDelta, true);
    }

    public EarlyStopping(int patience, double minDelta, boolean restoreBestWeights) {
        this.patience = Math.max(1, patience);
        this.minDelta = Math.max(0, minDelta);
        this.restoreBestWeights = restoreBestWeights;
    }

    public boolean shouldStop(double validationScore, Network network) {
        if (validationScore < bestScore - minDelta) {
            bestScore = validationScore;
            counter = 0;
            if (restoreBestWeights) {
                bestWeights = network.getParameters();
            }
            return false;
        } else {
            counter++;
            if (counter >= patience) {
                stopped = true;
                if (restoreBestWeights && bestWeights != null) {
                    network.setParameters(bestWeights);
                }
                return true;
            }
            return false;
        }
    }

    public void reset() {
        counter = 0;
        bestScore = Double.POSITIVE_INFINITY;
        bestWeights = null;
        stopped = false;
    }

    public int getCounter() { return counter; }
    public int getPatience() { return patience; }
    public double getBestScore() { return bestScore; }
    public boolean isStopped() { return stopped; }
}