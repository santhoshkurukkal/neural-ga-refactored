# Setup Guide for Neural-GA Refactored

## Prerequisites Installation

### Windows (PowerShell as Administrator)

```powershell
# Install Java 17 (Temurin)
winget install EclipseAdoptium.Temurin.17.JDK

# Install Maven
winget install Apache.Maven

# Install Git (if not already)
winget install Git.Git

# Verify installations
java -version
mvn -version
git --version
```

### Alternative: Manual Installation

1. **Java 17**: Download from https://adoptium.net/temurin/releases/?version=17
2. **Maven**: Download from https://maven.apache.org/download.cgi
3. **Git**: Download from https://git-scm.com/download/win

Add to PATH:
- `JAVA_HOME` = JDK installation directory (e.g., `C:\Program Files\Eclipse Adoptium\jdk-17.0.x.x-hotspot`)
- `MAVEN_HOME` = Maven installation directory
- Add `%JAVA_HOME%\bin`, `%MAVEN_HOME%\bin` to PATH

---

## Project Setup

### 1. Clone Repository

```powershell
# Create your GitHub repo first at github.com, then:
git clone https://github.com/YOUR_USERNAME/neural-ga-refactored.git
cd neural-ga-refactored
```

### 2. Build Project

```powershell
# Using Maven Wrapper (no Maven install required)
.\mvnw.cmd clean install -DskipTests

# Or with installed Maven
mvn clean install -DskipTests
```

### 3. Run Tests

```powershell
# All tests
.\mvnw.cmd test

# Specific module
.\mvnw.cmd test -pl data-pipeline
.\mvnw.cmd test -pl neural-core
.\mvnw.cmd test -pl ga-core
```

---

## Running the Application

### Training

```powershell
# Train on CPU data (included sample)
.\mvnw.cmd -pl cli exec:java -Dexec.mainClass="com.neuralga.cli.NeuralGAMain" -Dexec.args="train --data data/sample/cpu.txt --config config/training.yaml --output models/best-model.json"

# Train on synthetic data
.\mvnw.cmd -pl cli exec:java -Dexec.mainClass="com.neuralga.cli.NeuralGAMain" -Dexec.args="generate-data --type SINE_WITH_NOISE --length 1000 --output data/sample/synthetic.csv"
.\mvnw.cmd -pl cli exec:java -Dexec.mainClass="com.neuralga.cli.NeuralGAMain" -Dexec.args="train --data data/sample/synthetic.csv --output models/synthetic-model.json"
```

### Prediction

```powershell
# Batch prediction
.\mvnw.cmd -pl cli exec:java -Dexec.mainClass="com.neuralga.cli.NeuralGAMain" -Dexec.args="predict --model models/best-model.json --data data/sample/cpu.txt --output predictions.csv"
```

### Evaluation

```powershell
# Evaluate model
.\mvnw.cmd -pl cli exec:java -Dexec.mainClass="com.neuralga.cli.NeuralGAMain" -Dexec.args="evaluate --model models/best-model.json --data data/sample/cpu.txt"
```

### JavaFX GUI

```powershell
# Run GUI via Maven
.\mvnw.cmd -pl gui-javafx javafx:run

# Or build and run JAR
.\mvnw.cmd -pl gui-javafx package
java -jar gui-javafx/target/gui-javafx-1.0.0-SNAPSHOT.jar
```

---

## Project Structure

```
neural-ga-refactored/
├── pom.xml                          # Parent POM
├── README.md                        # Project overview
├── SETUP.md                         # This file
├── .github/workflows/ci.yml         # GitHub Actions CI
├── .mvn/wrapper/                    # Maven wrapper
├── mvnw.cmd                         # Maven wrapper (Windows)
├── mvnw                             # Maven wrapper (Linux/Mac)
├── config/                          # Configuration module
│   ├── pom.xml
│   ├── src/main/java/com/neuralga/config/
│   │   ├── AppConfig.java
│   │   ├── TrainingConfig.java
│   │   ├── GAConfig.java
│   │   ├── CascadeConfig.java
│   │   └── ConfigLoader.java
│   ├── src/main/resources/
│   │   ├── training.yaml            # Main config
│   │   └── logback.xml
│   └── src/test/...
├── data-pipeline/                   # Data handling
│   ├── pom.xml
│   ├── src/main/java/com/neuralga/data/
│   │   ├── CsvTimeseriesReader.java
│   │   ├── SyntheticDataGenerator.java
│   │   ├── TrainValTestSplitter.java
│   │   ├── DataNormalizer.java
│   │   ├── SlidingWindowDataset.java
│   │   └── DataPipeline.java
│   └── src/test/...
├── neural-core/                     # Neural network
│   ├── pom.xml
│   ├── src/main/java/com/neuralga/neural/
│   │   ├── ActivationFunction.java
│   │   ├── Neuron.java
│   │   ├── Layer.java
│   │   ├── Network.java
│   │   ├── CascadeCorrelationNetwork.java
│   │   ├── BackpropTrainer.java
│   │   ├── Optimizer.java
│   │   ├── LossFunction.java
│   │   ├── EarlyStopping.java
│   │   └── NetworkSerializer.java
│   └── src/test/...
├── ga-core/                         # Genetic Algorithm
│   ├── pom.xml
│   ├── src/main/java/com/neuralga/ga/
│   │   ├── Chromosome.java
│   │   ├── Population.java
│   │   ├── FitnessFunction.java
│   │   ├── SelectionStrategy.java
│   │   ├── CrossoverOperator.java
│   │   ├── MutationOperator.java
│   │   └── GeneticAlgorithm.java
│   └── src/test/...
├── training-pipeline/               # Cascade + GA integration
│   ├── pom.xml
│   ├── src/main/java/com/neuralga/pipeline/
│   │   ├── TrainingPipeline.java
│   │   ├── CascadeGATrainer.java
│   │   ├── ModelCheckpoint.java
│   │   └── ValidationEvaluator.java
│   └── src/test/...
├── prediction-service/              # Inference
│   ├── pom.xml
│   ├── src/main/java/com/neuralga/service/
│   │   ├── ModelLoader.java
│   │   └── PredictionService.java
│   └── src/test/...
├── cli/                             # Command-line
│   ├── pom.xml
│   ├── src/main/java/com/neuralga/cli/
│   │   └── NeuralGAMain.java
│   └── src/test/...
├── gui-javafx/                      # JavaFX GUI
│   ├── pom.xml
│   ├── src/main/java/com/neuralga/gui/
│   │   ├── NeuralGAApp.java
│   │   └── controller/
│   │       ├── MainController.java
│   │       ├── TrainingController.java
│   │       └── PredictionController.java
│   ├── src/main/resources/fxml/
│   │   ├── MainView.fxml
│   │   ├── TrainingView.fxml
│   │   └── PredictionView.fxml
│   └── src/main/resources/css/style.css
└── data/sample/
    ├── cpu.txt                      # Sample CPU data
    └── synthetic.csv                # Generated data
```

---

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
  output_activation: LINEAR

training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.01
  optimizer: ADAM
  loss: MSE
  early_stopping:
    patience: 10
    min_delta: 1e-4
  validation_split: 0.2
  test_split: 0.1

cascade:
  error_threshold: 0.5
  max_cascade_iterations: 10
  retrain_epochs_after_growth: 50
  fine_tune_learning_rate_factor: 0.1

ga:
  population_size: 50
  generations: 100
  elitism_count: 2
  tournament_size: 3
  crossover_rate: 0.8
  mutation_rate: 0.1
  crossover_operator: SBX
  mutation_operator: GAUSSIAN
  fitness_metric: VAL_MSE
```

---

## Key Features

| Feature | Implementation |
|---------|---------------|
| **Cascade Correlation** | Dynamic hidden layer growth via GA |
| **Genetic Algorithm** | SBX crossover, Gaussian mutation, Tournament selection |
| **Neural Network** | Tanh/ReLU/Sigmoid/Linear activations, Adam/SGD/RMSprop |
| **Data Pipeline** | CSV reader, Synthetic generator (7 types), Chronological split |
| **Normalization** | Z-Score, Min-Max, Robust, Auto-detect |
| **Serialization** | JSON (Jackson) with full architecture |
| **GUI** | JavaFX with FXML, background training |
| **Build** | Maven multi-module, wrapper, CI/CD |

---

## Synthetic Data Types

```bash
# Available types:
# SINE, SINE_WITH_NOISE, COSINE, SQUARE_WAVE, ARMA, MACKEY_GLASS, RANDOM_WALK

.\mvnw.cmd -pl cli exec:java -Dexec.mainClass="com.neuralga.cli.NeuralGAMain" \
  -Dexec.args="generate-data --type MACKEY_GLASS --length 2000 --noise 0.02 --output data/sample/chaos.csv"
```

---

## Model Format (JSON)

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "network": { "layers": [...] },
  "normalizer": { "method": "Z_SCORE", "mean": 0.5, "std": 0.2 },
  "config": { "input_window": 5, ... },
  "metrics": { "valMSE": 0.001, "testMSE": 0.002, "hiddenLayers": 3 }
}
```

---

## Troubleshooting

### Build Fails

```powershell
# Clean and rebuild
.\mvnw.cmd clean install -DskipTests

# Skip specific module tests
.\mvnw.cmd install -DskipTests -pl '!gui-javafx'
```

### JavaFX Not Found

```powershell
# Run with explicit JavaFX modules
java --module-path "%USERPROFILE%\.m2\repository\org\openjfx" --add-modules javafx.controls,javafx.fxml -jar gui-javafx/target/gui-javafx-1.0.0-SNAPSHOT.jar
```

### Out of Memory

```powershell
# Increase heap
set MAVEN_OPTS=-Xmx2g
.\mvnw.cmd install
```

---

## GitHub Push

```powershell
# After making changes
git add .
git commit -m "Your commit message"
git push origin main

# Tag release
git tag v1.0.0
git push origin v1.0.0
```

---

## Development

### Adding New Module

1. Create directory: `mkdir new-module/src/main/java/com/neuralga/newmodule`
2. Create `pom.xml` with parent reference
3. Add module to parent `pom.xml` `<modules>`
4. Add dependencies as needed
5. Run `.\mvnw.cmd install`

### Running Specific Tests

```powershell
# Single test class
.\mvnw.cmd test -Dtest=NeuronTest -pl neural-core

# With pattern
.\mvnw.cmd test -Dtest="*Test" -pl ga-core
```

---

## License

Educational/Research use. Original project by S. Santhosh Kumar, B. Sivanesan (2011).
Refactored 2024 with modern architecture.