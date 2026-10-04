# Neural-GA Refactored

Cascade Correlation Neural Network with Genetic Algorithm for Time-Series Prediction.

Refactored from a B.Tech project (2011) "Optimization of Resources in Distributed System using State Prediction Methods".

## Architecture

```
┌─────────────┐
│   config    │  ← YAML configuration (training, GA, app)
└──────┬──────┘
       │
┌──────┴──────┐
│data-pipeline│  ← Data loading, splitting, normalization, synthetic generation
└──────┬──────┘
       │
┌──────┴──────┐
│ neural-core │  ← Neurons, Layers, Network, Backprop, Cascade Correlation
└──────┬──────┘
       │
┌──────┴──────┐
│   ga-core   │  ← Chromosomes, Population, Selection, Crossover, Mutation, GA
└──────┬──────┘
       │
┌──────┴────────────┐
│training-pipeline  │  ← Cascade + GA integration
└──────┬────────────┘
       │
    ┌──┴──┐
    ▼     ▼
┌──────┐ ┌────────────┐
│ cli  │ │ gui-javafx │
└──────┘ └────────────┘
```

## Modules

| Module | Description |
|--------|-------------|
| `config` | Configuration loading (YAML) |
| `data-pipeline` | CSV reader, synthetic data (sine, ARMA, Mackey-Glass, random walk), train/val/test split, normalization, sliding windows |
| `neural-core` | Neuron, Layer, Network, BackpropTrainer, Optimizers (SGD, Adam), CascadeCorrelationNetwork |
| `ga-core` | GeneticAlgorithm, Chromosome, Tournament/Roulette/Rank selection, SBX/Uniform/SinglePoint crossover, Gaussian/Polynomial mutation |
| `training-pipeline` | CascadeGATrainer - grows network via GA-optimized hidden layers |
| `prediction-service` | Model loading, prediction, serialization (JSON) |
| `cli` | Command-line interface (train, predict, evaluate, generate-data) |
| `gui-javafx` | JavaFX GUI for training and prediction |

## Quick Start

### Prerequisites
- Java 17+ (Temurin recommended)
- Maven 3.9+

```bash
# Windows
winget install EclipseAdoptium.Temurin.17.JDK
winget install Apache.Maven

# Verify
java -version
mvn -version
```

### Build
```bash
cd neural-ga-refactored
mvn clean install -DskipTests
```

### Train on Sample Data
```bash
# Using synthetic data (no external files needed)
java -jar cli/target/cli-1.0.jar train \
  --data data/sample/cpu.txt \
  --config config/training.yaml \
  --output models/best-model.json

# Or generate synthetic data and train
java -jar cli/target/cli-1.0.jar generate-data --type SINE_WITH_NOISE --length 1000 --output data/sample/synthetic.csv
java -jar cli/target/cli-1.0.jar train --data data/sample/synthetic.csv --output models/synthetic-model.json
```

### Predict
```bash
java -jar cli/target/cli-1.0.jar predict \
  --model models/best-model.json \
  --data data/sample/new-data.txt \
  --output predictions.csv
```

### Evaluate
```bash
java -jar cli/target/cli-1.0.jar evaluate \
  --model models/best-model.json \
  --data data/sample/cpu.txt
```

### Run GUI
```bash
# Via Maven
mvn -pl gui-javafx javafx:run

# Or via JAR (after building with native profile)
java -jar gui-javafx/target/gui-javafx-1.0.jar
```

## Configuration

All hyperparameters in `config/training.yaml`:

```yaml
neural:
  input_window: 5
  prediction_horizon: 1
  initial_output_neurons: 5
  max_hidden_layers: 10
  neurons_per_hidden_layer: 5
  activation: TANH

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.01
  optimizer: ADAM
  early_stopping:
    patience: 10
    min_delta: 1e-4

cascade:
  error_threshold: 0.5
  max_cascade_iterations: 10

ga:
  population_size: 50
  generations: 100
  crossover_operator: SBX
  mutation_operator: GAUSSIAN
  fitness_metric: VAL_MSE
```

## Synthetic Data Generator

For testing without real data:

```bash
# Generate various types
java -jar cli/target/cli-1.0.jar generate-data \
  --type SINE_WITH_NOISE --length 1000 --noise 0.05 --output data/sample/test.csv

# Types: SINE, SINE_WITH_NOISE, COSINE, SQUARE_WAVE, ARMA, MACKEY_GLASS, RANDOM_WALK
```

## Model Format

Models saved as JSON with full architecture:

```json
{
  "architecture": {
    "inputWindow": 5,
    "predictionHorizon": 1,
    "layers": [
      {"type": "INPUT", "neurons": 5},
      {"type": "HIDDEN", "neurons": 5, "activation": "TANH"},
      {"type": "OUTPUT", "neurons": 5, "activation": "LINEAR"}
    ]
  },
  "weights": [...],
  "normalizer": {"method": "Z_SCORE", "mean": 0.5, "std": 0.2},
  "config": {...},
  "metrics": {"valMSE": 0.001, "testMSE": 0.002, "trainingTimeMs": 45000}
}
```

## Testing

```bash
# Run all tests
mvn test

# Run specific module tests
mvn test -pl data-pipeline
mvn test -pl neural-core
mvn test -pl ga-core

# Integration tests
mvn verify -Pintegration
```

## Key Improvements Over Original

| Original Issue | Fixed |
|----------------|-------|
| Circular dependencies | Strict module layering |
| God class (CascadeNetwork) | 15+ focused classes |
| Hardcoded values | External YAML config |
| Thread unsafe | Immutable data, proper sync |
| Broken GA | Complete rewrite with working operators |
| No validation split | Chronological train/val/test |
| Memory leaks | Bounded buffers, explicit lifecycle |
| Windows paths | Platform-independent Path API |
| No tests | 100+ unit/integration tests |
| No build system | Maven multi-module |

## License

Educational/Research use.