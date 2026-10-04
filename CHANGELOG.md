# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-01-15

### Added - Complete Refactoring

#### Architecture
- **Maven multi-module structure** with 8 modules: config, data-pipeline, neural-core, ga-core, training-pipeline, prediction-service, cli, gui-javafx
- **Strict dependency layering** - no circular dependencies
- **Configuration-driven** - all hyperparameters externalized to YAML

#### Configuration Module
- `TrainingConfig` - Neural network and training parameters
- `GAConfig` - Genetic algorithm parameters with termination conditions
- `CascadeConfig` - Cascade correlation growth parameters
- `AppConfig` - Application paths, logging, random seed
- `ConfigLoader` - YAML loading with defaults fallback
- Comprehensive unit tests

#### Data Pipeline Module
- `CsvTimeseriesReader` - Robust CSV/time-series loading with header/comment support
- `SyntheticDataGenerator` - 7 data types: SINE, SINE_WITH_NOISE, COSINE, SQUARE_WAVE, ARMA, MACKEY_GLASS, RANDOM_WALK
- `TrainValTestSplitter` - Chronological splitting (no random shuffle for time-series)
- `DataNormalizer` - Z-Score, Min-Max, Robust, Auto-detect methods
- `SlidingWindowDataset` - Configurable window/horizon/stride
- `DataPipeline` - Orchestrates full data preparation flow
- Full test coverage

#### Neural Core Module
- `ActivationFunction` enum: TANH, SIGMOID, RELU, LEAKY_RELU, LINEAR with derivatives
- `Neuron` - Xavier initialization, forward/backward pass
- `Layer` - Vectorized forward/backward, weight matrix operations
- `Network` - Multi-layer forward/backward, parameter serialization
- `CascadeCorrelationNetwork` - Dynamic hidden layer addition with weight preservation
- `BackpropTrainer` - Mini-batch training, early stopping, multiple optimizers
- `Optimizer` - SGD, Adam (with bias correction), RMSprop
- `LossFunction` - MSE, MAE, Huber with gradients
- `EarlyStopping` - Patience, min_delta, best weights restoration
- `NetworkSerializer` - JSON serialization with Jackson

#### GA Core Module
- `Chromosome` - Double array genes, fitness, mutation
- `Population` - Statistics, tournament/roulette/rank selection
- `SelectionStrategy` - Tournament, Roulette Wheel, Rank-based
- `CrossoverOperator` - SBX (Simulated Binary), Uniform, Single-Point
- `MutationOperator` - Gaussian, Polynomial
- `GeneticAlgorithm` - Elitism, parallel fitness evaluation, listeners, termination conditions
- Full operator test coverage

#### Training Pipeline Module
- `CascadeGATrainer` - Complete cascade correlation with GA-optimized hidden layers
  - Phase 1: Train output-only network
  - Phase 2: GA optimizes new hidden layer weights
  - Phase 3: Add layer, fine-tune with reduced LR
  - Repeat until threshold/max layers reached
- `TrainingPipeline` - High-level orchestration with data pipeline integration
- `ModelCheckpoint` - Periodic best-model saving
- `ValidationEvaluator` - MSE, MAE, R² metrics on train/val/test

#### Prediction Service Module
- `ModelLoader` - Load JSON models with network, normalizer, config, metrics
- `PredictionService` - Single/batch/sequence prediction with history management
- `BatchPredictor` - Dataset prediction with CSV export

#### CLI Module
- Picocli-based commands:
  - `train` - Train model with config file
  - `predict` - Batch/interactive prediction
  - `evaluate` - Model evaluation with metrics
  - `generate-data` - Synthetic data generation
- Maven Shade plugin for fat JAR

#### GUI Module (JavaFX)
- `NeuralGAApp` - Main application
- `MainView` - Tabbed interface (Training/Prediction)
- `TrainingView` - Config UI, data selection, progress, logs
- `PredictionView` - Model loading, batch/interactive prediction
- FXML-based views with CSS styling
- Background training with Task/ProgressBar
- Maven JavaFX plugin for `mvn javafx:run`

#### Infrastructure
- GitHub Actions CI (build, test, checkstyle, SpotBugs, native image)
- Maven Wrapper (no system Maven required)
- Comprehensive `.gitignore`
- Sample CPU data (500 points)
- Full documentation (README, SETUP, CHANGELOG)

### Fixed (vs Original 2011 Code)

| Original Issue | Fix |
|---------------|-----|
| Circular dependencies (Neural ↔ GA) | Strict module layering |
| God class (CascadeNetwork 346 lines) | 15+ focused classes |
| Hardcoded values (E://cpu.txt, 5, 6, 2.7, 90) | External YAML config |
| Thread unsafe (shared Vectors) | Immutable data, proper sync |
| Broken GA (stubs for Mutation/Crossover) | Complete GA implementation |
| No validation split | Chronological train/val/test (70/20/10) |
| Memory leaks (growing Vectors) | Bounded buffers, explicit lifecycle |
| Windows-specific paths | Platform-independent Path API |
| No tests | 50+ unit/integration tests |
| No build system | Maven multi-module with wrapper |
| No serialization | JSON model export/import |

### Changed

- Language: Java 17 (from Java 6/7 era)
- Build: Maven (from BlueJ project)
- GUI: JavaFX (from Swing)
- Config: YAML (from hardcoded)
- Logging: SLF4J/Logback (from System.out.println)
- Math: Custom MatrixOps (no external deps)

---

## Migration from Original

Original structure:
```
Project/Main Project/
├── Neural/ (CascadeNetwork, neurons, Input)
├── GA/ (Chromosome, Fitness, Selection, etc.)
```

New structure:
```
neural-ga-refactored/
├── config/           # All configuration
├── data-pipeline/    # Data loading, splitting, normalization
├── neural-core/      # Neural network components
├── ga-core/          # Genetic algorithm components
├── training-pipeline/# Cascade + GA integration
├── prediction-service/# Model loading, prediction
├── cli/              # Command-line interface
└── gui-javafx/       # JavaFX GUI
```

---

## Future Roadmap

- [ ] Python bindings via JPype/GraalVM
- [ ] REST API server module
- [ ] Hyperparameter optimization (Optuna integration)
- [ ] Distributed GA (island model)
- [ ] ONNX export for model portability
- [ ] Web UI (Spring Boot + React)