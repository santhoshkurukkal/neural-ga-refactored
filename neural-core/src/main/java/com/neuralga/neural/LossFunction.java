package com.neuralga.neural;

public enum LossFunction {
    MSE {
        @Override
        public double compute(double[] predicted, double[] target) {
            if (predicted.length != target.length) {
                throw new IllegalArgumentException("Length mismatch");
            }
            double sum = 0;
            for (int i = 0; i < predicted.length; i++) {
                double diff = predicted[i] - target[i];
                sum += diff * diff;
            }
            return sum / predicted.length;
        }

        @Override
        public double[] gradient(double[] predicted, double[] target) {
            double[] grad = new double[predicted.length];
            for (int i = 0; i < predicted.length; i++) {
                grad[i] = 2.0 * (predicted[i] - target[i]) / predicted.length;
            }
            return grad;
        }
    },
    MAE {
        @Override
        public double compute(double[] predicted, double[] target) {
            double sum = 0;
            for (int i = 0; i < predicted.length; i++) {
                sum += Math.abs(predicted[i] - target[i]);
            }
            return sum / predicted.length;
        }

        @Override
        public double[] gradient(double[] predicted, double[] target) {
            double[] grad = new double[predicted.length];
            for (int i = 0; i < predicted.length; i++) {
                double diff = predicted[i] - target[i];
                grad[i] = diff > 0 ? 1.0 : (diff < 0 ? -1.0 : 0.0);
            }
            return grad;
        }
    },
    HUBER {
        private static final double DELTA = 1.0;

        @Override
        public double compute(double[] predicted, double[] target) {
            double sum = 0;
            for (int i = 0; i < predicted.length; i++) {
                double diff = Math.abs(predicted[i] - target[i]);
                if (diff <= DELTA) {
                    sum += 0.5 * diff * diff;
                } else {
                    sum += DELTA * (diff - 0.5 * DELTA);
                }
            }
            return sum / predicted.length;
        }

        @Override
        public double[] gradient(double[] predicted, double[] target) {
            double[] grad = new double[predicted.length];
            for (int i = 0; i < predicted.length; i++) {
                double diff = predicted[i] - target[i];
                if (Math.abs(diff) <= DELTA) {
                    grad[i] = diff;
                } else {
                    grad[i] = DELTA * (diff > 0 ? 1.0 : -1.0);
                }
            }
            return grad;
        }
    };

    public abstract double compute(double[] predicted, double[] target);
    public abstract double[] gradient(double[] predicted, double[] target);

    public static LossFunction fromString(String name) {
        if (name == null) return MSE;
        return switch (name.toUpperCase()) {
            case "MSE" -> MSE;
            case "MAE" -> MAE;
            case "HUBER" -> HUBER;
            default -> MSE;
        };
    }
}