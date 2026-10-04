# Neural-GA Refactored

**Cascade Correlation Neural Network with Genetic Algorithm for Time-Series Prediction**

Refactored from a B.Tech project (2011) "Optimization of Resources in Distributed System using State Prediction Methods" by S. Santhosh Kumar, B. Sivanesan.

## Project Status

✅ **Build**: All modules compile successfully  
✅ **Tests**: 170+ unit/integration tests pass  
✅ **Core Training**: Initial network training works (early stopping, backprop, optimizers)  
✅ **Data Pipeline**: CSV loading, synthetic generation (7 types), splitting, normalization  
✅ **Genetic Algorithm**: Chromosomes, selection, crossover, mutation, parallel evaluation  
✅ **Serialization**: JSON model save/load  
✅ **CLI**: Train, predict, evaluate, generate-data commands  
⚠️ **GA Cascade**: Bug in fitness evaluation (output layer size mismatch during hidden layer addition)

## Architecture

```
neural-ga-refactored/
├── config/                 # YAML configuration (training, GA, cascade, app)
├── data-pipeline/          # CSV reader, synthetic data, splitting, normalization
├── neural-core/            # Neurons, layers, networks, backprop, cascade correlation
├── ga-core/                # Chromosomes, population, GA operators, algorithm
├── training-pipeline/      # Cascade + GA integration
├── prediction-service/     # Model loading, prediction, JSON serialization
├── cli/                    # Command-line interface (train, predict, evaluate, generate)
├── gui-javafx/             # JavaFX GUI (training/prediction tabs)
└── data/sample/            # Sample CPU data + synthetic generator
```

## Quick Start

### Prerequisites
```powershell
# Install Java 17 + Maven
winget install EclipseAdoptium.Temurin.17.JDK
winget install Apache.Maven
winget install Git.Git
```

### Build & Test
```powershell
cd neural-ga-refactored
$env:JAVA_HOME = "C:\Program Files\Eclipse Adoptium\jdk-17.0.20.101-hotspot"
$env:Path += ";$env:JAVA_HOME\bin"
.\mvnw.cmd clean install -DskipTests
.\mvnw.cmd test
```

### Generate Synthetic Data
```powershell
java -jar cli/target/cli-1.0.0-SNAPSHOT.jar generate-data --type SINE_WITH_NOISE --length 1000 --output data/sample/synthetic.csv
```

### Train Model (Basic - No Cascade)
```powershell
java -jar cli/target/cli-1.0.0-SNAPSHOT.jar train --data data/sample/synthetic.csv --output models/model.json
```

### Train Model (With Cascade - Currently Has Bug)
```powershell
java -jar cli/target/cli-1.0.0-SNAPSHOT.jar train --data data/sample/synthetic.csv --output models/model.json
# Note: GA cascade has bug in fitness evaluation (output layer size mismatch)
```

### Predict
```powershell
java -jar cli/target/cli-1.0.0-SNAPSHOT.jar predict --model models/model.json --data data/sample/synthetic.csv --output predictions.csv
```

### Evaluate
```powershell
java -jar cli/target/cli-1.0.0-SNAPSHOT.jar evaluate --model models/model.json --data data/sample/synthetic.csv
```

### Run GUI
```powershell
.\mvnw.cmd -pl gui-javafx javafx:run
```

## Configuration

All hyperparameters in `config/training.yaml`:
```yaml
neural:
  input_window: 5
  prediction_horizon: 1
  initial_output_neurons: 1  # = prediction_horizon
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

## Known Issues

1. **GA Cascade Bug**: Fitness evaluation fails with "Index 1 out of bounds for length 1" - output layer size mismatch during hidden layer addition in GA fitness evaluation. The output layer should have `prediction_horizon` neurons (1) but appears to have more during GA fitness evaluation.

2. **Config File**: CLI looks for `config/training.yaml` in working directory. Create one or use defaults.

3. **SpotBugs Plugin**: Version 4.8.6 not found in Maven Central (use 4.8.5 or remove).

## GitHub Setup

```bash
git init
git add .
git commit -m "Initial commit: Neural-GA refactored with Maven multi-module architecture"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/neural-ga-refactored.git
git push -u origin main
```

## CI/CD

GitHub Actions workflow at `.github/workflows/ci.yml`:
- Build & test on Ubuntu
- Code quality (Checkstyle, SpotBugs)
- Native image build for GUI (on main branch)

## License

Educational/Research use. Original project by S. Santhosh Kumar, B. Sivanesan (2011).