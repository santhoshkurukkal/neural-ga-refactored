package com.neuralga.neural;

public enum ActivationFunction {
    TANH {
        @Override
        public double activate(double x) {
            return Math.tanh(x);
        }

        @Override
        public double derivative(double x) {
            double t = Math.tanh(x);
            return 1.0 - t * t;
        }
    },
    SIGMOID {
        @Override
        public double activate(double x) {
            return 1.0 / (1.0 + Math.exp(-x));
        }

        @Override
        public double derivative(double x) {
            double s = activate(x);
            return s * (1.0 - s);
        }
    },
    RELU {
        @Override
        public double activate(double x) {
            return Math.max(0, x);
        }

        @Override
        public double derivative(double x) {
            return x > 0 ? 1.0 : 0.0;
        }
    },
    LEAKY_RELU {
        @Override
        public double activate(double x) {
            return x > 0 ? x : 0.01 * x;
        }

        @Override
        public double derivative(double x) {
            return x > 0 ? 1.0 : 0.01;
        }
    },
    LINEAR {
        @Override
        public double activate(double x) {
            return x;
        }

        @Override
        public double derivative(double x) {
            return 1.0;
        }
    };

    public abstract double activate(double x);
    public abstract double derivative(double x);

    public static ActivationFunction fromString(String name) {
        if (name == null) return LINEAR;
        return switch (name.toUpperCase()) {
            case "TANH" -> TANH;
            case "SIGMOID" -> SIGMOID;
            case "RELU" -> RELU;
            case "LEAKY_RELU" -> LEAKY_RELU;
            case "LINEAR" -> LINEAR;
            default -> {
                System.err.println("Unknown activation: " + name + ", defaulting to LINEAR");
                yield LINEAR;
            }
        };
    }
}