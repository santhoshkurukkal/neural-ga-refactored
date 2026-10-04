# Neural-GA Refactored - Initial Project Plan

## 1. Project Overview

### 1.1 What This Project Is
Neural-GA Refactored is a modern Java implementation of a **Cascade Correlation Neural Network with Genetic Algorithm (GA) optimization** for time-series prediction. The system dynamically grows neural network architecture by adding hidden layers whose weights are optimized via a genetic algorithm, enabling automatic architecture discovery for temporal forecasting tasks.

### 1.2 Origin: B.Tech 2011 Project
This project is a **complete refactoring** of the original B.Tech final year project (2011):
- **Title**: "Optimization of Resources in Distributed System using State Prediction Methods"
- **Authors**: S. Santhosh Kumar, B. Sivanesan
- **Original Stack**: Java 6/7, BlueJ IDE, Swing GUI, no build system
- **Original Code Location**: `Project/Main Project/` directory with two packages:
  - `Neural/` - CascadeNetwork, neurons, Input classes
  - `GA/` - Chromosome, Fitness, Selection, Crossover, Mutation, Genetics classes

### 1.3 Refactoring Goals
| Goal | Original State | Target State |
|------|----------------|--------------|
| **Build System** | None (BlueJ project) | Maven multi-module with wrapper |
| **Architecture** | Circular dependencies, God classes | Strict layering, 15+ focused classes |
| **Configuration** | Hardcoded values (paths, hyperparameters) | External YAML configuration |
| **Thread Safety** | Shared mutable Vectors | Immutable data, proper synchronization |
| **Genetic Algorithm** | Stub implementations | Complete GA with SBX, Gaussian mutation, tournament/roulette/rank selection |
| **Data Pipeline** | Manual CSV parsing | Robust pipeline: CSV, synthetic (7 types), splitting, normalization, sliding windows |
| **Validation** | No validation split | Chronological train/val/test (70/20/10) |
| **Serialization** | None | JSON model export/import with full architecture |
| **Testing** | Zero tests | 170+ unit/integration tests |
| **GUI** | Swing | JavaFX with FXML/CSS, background tasks |
| **Logging** | System.out.println | SLF4J/Logback |
| **Platform** | Windows-specific paths | Platform-independent Path API |

---

## 2. Technical Architecture

### 2.1 Module Breakdown (8 Maven Modules)

```
neural-ga-refactored/
├── config/                 # YAML configuration (training, GA, cascade, app)
├── data-pipeline/          # CSV reader, synthetic data, splitting, normalization, sliding windows
├── neural-core/            # Neurons, layers, networks, backprop, cascade correlation, optimizers
├── ga-core/                # Chromosomes, population, selection, crossover, mutation, GA algorithm
├── training-pipeline/      # Cascade + GA integration (CascadeGATrainer)
├── prediction-service/     # Model loading, prediction, JSON serialization
├── cli/                    # Command-line interface (Picocli)
└── gui-javafx/             # JavaFX GUI (FXML, CSS, background training)
```

### 2.2 Dependency Graph (Strict Layering)

```
config (no deps)
    ▲
    │
data-pipeline ──────────► config
    ▲
    │
neural-core ◄────────────┤
    ▲                    │
    │                    │
ga-core ◄────────────────┤
    ▲                    │
    │                    │
training-pipeline ◄──────┼────► neural-core, ga-core, data-pipeline
    ▲                    │
    │                    │
prediction-service ◄─────┘
    ▲
    │
cli ◄────────────────────┼────► training-pipeline, prediction-service, config
    │
    │
gui-javafx ◄─────────────┘
```

**Key Rule**: Dependencies flow **downward only** (config → data-pipeline → neural-core/ga-core → training-pipeline → prediction-service → cli/gui). No circular dependencies.

### 2.3 Data Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  Raw Data   │────►│ DataPipeline│────►│  Training   │────►│  Trained    │
│  (CSV/Synth)│     │ (split,     │     │  Pipeline   │     │  Model      │
│             │     │  normalize, │     │ (Cascade+GA)│     │  (JSON)     │
└─────────────┘     │  windows)   │     └─────────────┘     └──────┬──────┘
                    └─────────────┘                                │
                          │                                        │
                          ▼                                        ▼
                   ┌─────────────┐                         ┌─────────────┐
                   │  Validation │                         │ Prediction  │
                   │  Evaluator  │                         │  Service    │
                   └─────────────┘                         └─────────────┘
```

### 2.4 Key Classes by Module

| Module | Key Classes |
|--------|-------------|
| **config** | `TrainingConfig`, `GAConfig`, `CascadeConfig`, `AppConfig`, `ConfigLoader` |
| **data-pipeline** | `CsvTimeseriesReader`, `SyntheticDataGenerator`, `TrainValTestSplitter`, `DataNormalizer`, `SlidingWindowDataset`, `DataPipeline` |
| **neural-core** | `Neuron`, `Layer`, `Network`, `CascadeCorrelationNetwork`, `BackpropTrainer`, `Optimizer` (SGD/Adam/RMSprop), `ActivationFunction`, `LossFunction`, `EarlyStopping`, `NetworkSerializer` |
| **ga-core** | `Chromosome`, `Population`, `FitnessFunction`, `SelectionStrategy` (Tournament/Roulette/Rank), `CrossoverOperator` (SBX/Uniform/SinglePoint), `MutationOperator` (Gaussian/Polynomial), `GeneticAlgorithm` |
| **training-pipeline** | `CascadeGATrainer`, `TrainingPipeline`, `ModelCheckpoint`, `ValidationEvaluator` |
| **prediction-service** | `ModelLoader`, `PredictionService`, `BatchPredictor` |
| **cli** | `NeuralGAMain` (Picocli: train, predict, evaluate, generate-data) |
| **gui-javafx** | `NeuralGAApp`, `MainController`, `TrainingController`, `PredictionController`, FXML views |

---

## 3. Implementation Phases

### Phase 1: Foundation & Configuration ✅
**Duration**: Initial setup
**Deliverables**:
- Maven multi-module project structure (parent POM + 8 modules)
- Maven Wrapper (`mvnw`/`mvnw.cmd`)
- Parent POM with dependency management (Jackson, SnakeYAML, SLF4J/Logback, JUnit 5, AssertJ, Mockito, Picocli, JavaFX)
- `config` module: YAML loading with SnakeYAML + Jackson, typed config classes, defaults fallback
- `training.yaml` with all hyperparameters externalized
- GitHub Actions CI workflow (build, test, checkstyle, SpotBugs, native image)

### Phase 2: Data Pipeline ✅
**Duration**: After config
**Deliverables**:
- `CsvTimeseriesReader`: Robust CSV parsing (headers, comments, flexible delimiters)
- `SyntheticDataGenerator`: 7 data types (SINE, SINE_WITH_NOISE, COSINE, SQUARE_WAVE, ARMA, MACKEY_GLASS, RANDOM_WALK)
- `TrainValTestSplitter`: Chronological splitting (no random shuffle for time-series)
- `DataNormalizer`: Z-Score, Min-Max, Robust, Auto-detect methods
- `SlidingWindowDataset`: Configurable window/horizon/stride for supervised learning
- `DataPipeline`: Orchestrates full flow → `PreparedData` with train/val/test datasets
- Comprehensive unit tests for all components

### Phase 3: Neural Core ✅
**Duration**: After data-pipeline
**Deliverables**:
- `ActivationFunction` enum: TANH, SIGMOID, RELU, LEAKY_RELU, LINEAR with derivatives
- `Neuron`: Xavier initialization, forward/backward pass
- `Layer`: Vectorized forward/backward, weight matrix operations (custom `MatrixOps`, no external deps)
- `Network`: Multi-layer forward/backward, parameter serialization
- `CascadeCorrelationNetwork`: Dynamic hidden layer addition with weight preservation
- `BackpropTrainer`: Mini-batch training, early stopping, multiple optimizers
- `Optimizer`: SGD, Adam (with bias correction), RMSprop
- `LossFunction`: MSE, MAE, Huber with gradients
- `EarlyStopping`: Patience, min_delta, best weights restoration
- `NetworkSerializer`: JSON serialization with Jackson (architecture, weights, normalizer, config, metrics)
- Full test coverage including integration tests

### Phase 4: GA Core ✅
**Duration**: After neural-core
**Deliverables**:
- `Chromosome`: Double array genes, fitness, mutation tracking
- `Population`: Statistics (min/max/mean/std fitness), elitism
- `SelectionStrategy`: Tournament (configurable size), Roulette Wheel, Rank-based
- `CrossoverOperator`: SBX (Simulated Binary Crossover), Uniform, Single-Point
- `MutationOperator`: Gaussian (with bounds), Polynomial
- `GeneticAlgorithm`: Elitism, parallel fitness evaluation (ExecutorService), listeners, termination conditions (generations, fitness threshold, stall)
- Full operator test coverage with statistical validation

### Phase 5: Training Pipeline ✅ (with known bug)
**Duration**: After neural-core + ga-core
**Deliverables**:
- `CascadeGATrainer`: Complete cascade correlation with GA-optimized hidden layers
  - Phase 1: Train output-only network (backprop)
  - Phase 2: GA optimizes new hidden layer weights (chromosome = input→hidden weights + biases)
  - Phase 3: Add layer, fine-tune with reduced learning rate
  - Repeat until error threshold / max layers / max iterations
- `TrainingPipeline`: High-level orchestration with data pipeline integration
- `ModelCheckpoint`: Periodic best-model saving
- `ValidationEvaluator`: MSE, MAE, R² metrics on train/val/test
- **Known Bug**: GA cascade fitness evaluation fails with "Index 1 out of bounds for length 1" - output layer size mismatch during hidden layer addition

### Phase 6: Prediction Service ✅
**Duration**: After training-pipeline
**Deliverables**:
- `ModelLoader`: Load JSON models with network, normalizer, config, metrics
- `PredictionService`: Single/batch/sequence prediction with history management
- `BatchPredictor`: Dataset prediction with CSV export

### Phase 7: CLI ✅
**Duration**: After prediction-service
**Deliverables**:
- `NeuralGAMain` with Picocli commands:
  - `train`: Train model with config file
  - `predict`: Batch/interactive prediction
  - `evaluate`: Model evaluation with metrics
  - `generate-data`: Synthetic data generation
- Maven Shade plugin for fat JAR (`cli-1.0.0-SNAPSHOT.jar`)

### Phase 8: GUI (JavaFX) ✅
**Duration**: After CLI
**Deliverables**:
- `NeuralGAApp`: Main JavaFX application
- `MainView`: Tabbed interface (Training / Prediction)
- `TrainingView`: Config UI, data selection, progress bars, live logs
- `PredictionView`: Model loading, batch/interactive prediction, results table
- FXML-based views with CSS styling
- Background training with `Task`/`ProgressBar` (non-blocking UI)
- Maven JavaFX plugin for `mvn javafx:run`
- Native image profile for distribution

---

## 4. Key Technical Decisions

### 4.1 Why Maven?
- **Standardized build** across environments (no IDE lock-in)
- **Multi-module support** for strict dependency layering
- **Dependency management** via `<dependencyManagement>` in parent POM
- **Rich plugin ecosystem** (shade, exec, javafx, checkstyle, spotbugs, javadoc)
- **Maven Wrapper** (`mvnw`) ensures reproducible builds without system Maven install
- **Industry standard** for Java projects

### 4.2 Why Java 17?
- **LTS release** (long-term support until 2029)
- **Modern language features**: Records, pattern matching, sealed classes, text blocks, switch expressions
- **Performance improvements** over Java 8/11 (GC, JIT, startup)
- **Strong ecosystem support** (Spring Boot 3+, Quarkus, Micronaut all require 17+)
- **Original project was Java 6/7** - massive upgrade enables modern practices

### 4.3 Why YAML Configuration?
- **Human-readable** and editable (vs XML/JSON)
- **Hierarchical structure** maps naturally to config objects
- **Comments supported** for documentation
- **SnakeYAML + Jackson** provides type-safe binding to POJOs
- **Externalizes all hyperparameters** - no recompilation for tuning
- **Environment-specific configs** possible (dev/test/prod)

### 4.4 Why Jackson (not Gson)?
- **Best-in-class YAML support** via `jackson-dataformat-yaml`
- **Java 8+ time API support** via `jackson-datatype-jsr310`
- **Annotation-driven** configuration (`@JsonProperty`, `@JsonCreator`)
- **Polymorphic type handling** for activation functions, optimizers, GA operators
- **Streaming API** for large models
- **Mature, well-maintained**, industry standard

### 4.5 Why Custom MatrixOps (No External Math Lib)?
- **Zero dependencies** - simpler deployment, smaller JARs
- **Full control** over numerical precision and performance
- **Educational value** - demonstrates understanding of linear algebra
- **Sufficient for this scale** - networks are small (5-50 neurons)
- **Avoids** ND4J, EJML, Apache Commons Math overhead

### 4.6 Why Picocli for CLI?
- **Annotation-driven** command definition (clean, declarative)
- **Subcommands** support (train/predict/evaluate/generate-data)
- **Auto-generated help** with colors
- **Type-safe** parameter binding
- **Shell completion** support (bash/zsh/fish)
- **Single-file** or shaded JAR friendly

### 4.7 Why JavaFX (not Swing)?
- **Modern UI toolkit** (CSS styling, FXML separation, property binding)
- **Scene Builder** support for visual design
- **Active development** (OpenJFX, Gluon)
- **Native packaging** via `jpackage` / GraalVM native-image
- **Original used Swing** - JavaFX is the modern successor

### 4.8 Why Chronological Split (Not Random)?
- **Time-series data** has temporal dependencies
- **Random shuffle leaks future into past** → invalid evaluation
- **Train (70%) → Val (20%) → Test (10%)** in chronological order
- **Walk-forward validation** would be next improvement

---

## 5. Known Issues & Technical Debt

### 5.1 Critical Bugs

| ID | Issue | Location | Severity | Status |
|----|-------|----------|----------|--------|
| **BUG-001** | GA Cascade fitness evaluation fails with "Index 1 out of bounds for length 1" | `CascadeGATrainer.addHiddenLayerWithGA()` line ~124-137 | **Critical** | Open |
| **BUG-002** | Config file not found if not in working directory | CLI looks for `config/training.yaml` relative to CWD | Medium | Open |

#### BUG-001 Details: GA Cascade Output Layer Size Mismatch
**Root Cause**: During GA fitness evaluation, a test network is cloned and a hidden layer is added via `testNet.addHiddenLayer(weights, biases)`. However, the `CascadeCorrelationNetwork` constructor initializes the output layer with `initialOutputNeurons` (from config, default 5), but the actual prediction horizon is 1. When the test network is created via `cloneNetwork()`, it creates a **new** `CascadeCorrelationNetwork` which re-initializes the output layer to 5 neurons instead of preserving the original network's output layer (1 neuron). The GA chromosome encoding assumes the output layer size matches the prediction horizon.

**Evidence**: 
- Config has `initial_output_neurons: 5` but `prediction_horizon: 1`
- `CascadeCorrelationNetwork` constructor uses `initialOutputNeurons` for output layer size
- Fitness evaluation tries to access output index 1 on a length-1 array

**Fix Options**:
1. Fix `cloneNetwork()` to copy output layer weights/biases from original
2. Make `CascadeCorrelationNetwork` constructor accept output layer size separately
3. Ensure `initialOutputNeurons` == `predictionHorizon` in config validation

### 5.2 Limitations

| Area | Limitation | Impact |
|------|------------|--------|
| **Cascade GA** | Broken (see BUG-001) | Cannot grow networks via GA; only basic backprop works |
| **Parallel GA** | Uses `ExecutorService` but no thread pool config | Resource contention on large populations |
| **Data Pipeline** | No streaming for large datasets | Memory-bound for datasets > RAM |
| **Normalization** | Fitted on train only (correct) but no persistence of normalizer params separate from model | Must save full model to persist normalizer |
| **Early Stopping** | Restores best weights but no callback/hooks | Limited extensibility |
| **Serialization** | JSON only, no binary format | Large models slow to save/load |
| **GUI** | No progress persistence across restarts | Long training lost if app closes |
| **Testing** | No integration test for full cascade+GA flow | Bug BUG-001 not caught by tests |

### 5.3 Technical Debt

| Item | Description | Effort |
|------|-------------|--------|
| **SpotBugs version** | `4.8.6` not in Maven Central (use 4.8.5) | Low |
| **Checkstyle config** | Not yet added (Google checks recommended) | Low |
| **Javadoc coverage** | ~60% public API documented | Medium |
| **Error handling** | Many `catch (Exception e)` returning `Double.NEGATIVE_INFINITY` | Medium |
| **Logging consistency** | Mix of `log.info` with `{}` placeholders and string concat | Low |
| **Magic numbers** | Some hardcoded values in neural-core (e.g., Xavier scale) | Low |
| **GA termination** | No convergence detection beyond stall generations | Medium |
| **Config validation** | No validation of YAML values (e.g., negative epochs) | Low |

---

## 6. Future Roadmap

### 6.1 Immediate (Fix Critical Bugs) - Priority: P0

| Task | Description | Estimate |
|------|-------------|----------|
| **FIX-001** | Fix GA Cascade output layer size mismatch in `CascadeGATrainer.cloneNetwork()` and/or `CascadeCorrelationNetwork` | 2-4 hrs |
| **FIX-002** | Add config file path resolution (classpath + CWD + explicit) | 1 hr |
| **FIX-003** | Update SpotBugs to 4.8.5 in parent POM | 15 min |
| **FIX-004** | Add integration test for cascade+GA to prevent regression | 4-8 hrs |

### 6.2 Short Term (Core Improvements) - Priority: P1

| Task | Description | Estimate |
|------|-------------|----------|
| **FEAT-001** | Add config validation (Bean Validation / custom) | 1-2 days |
| **FEAT-002** | Implement walk-forward validation for time-series | 2-3 days |
| **FEAT-003** | Add learning rate scheduling (cosine annealing, step decay) | 1-2 days |
| **FEAT-004** | Binary model serialization (Protocol Buffers / custom) | 2-3 days |
| **FEAT-005** | GA convergence detection (fitness variance, diversity metrics) | 1-2 days |
| **FEAT-006** | Thread pool configuration for parallel GA evaluation | 1 day |
| **FEAT-007** | Model checkpointing during cascade iterations | 1 day |

### 6.3 Medium Term (New Capabilities) - Priority: P2

| Task | Description | Estimate |
|------|-------------|----------|
| **FEAT-008** | **REST API Server Module** (Spring Boot / Helidon / Vert.x) | 1-2 weeks |
| **FEAT-009** | **Hyperparameter Optimization** (Optuna integration via JNI/Python) | 2-3 weeks |
| **FEAT-010** | **Distributed GA** (Island model with MPI/gRPC) | 2-4 weeks |
| **FEAT-011** | **ONNX Export** for model portability to Python/ONNX Runtime | 1-2 weeks |
| **FEAT-012** | Additional activation functions (GELU, Swish, Mish) | 1 week |
| **FEAT-013** | Bayesian Optimization for cascade hyperparameters | 2 weeks |

### 6.4 Long Term (Platform & Ecosystem) - Priority: P3

| Task | Description | Estimate |
|------|-------------|----------|
| **FEAT-014** | **Web UI** (Spring Boot + React/TypeScript) | 4-8 weeks |
| **FEAT-015** | **Python Bindings** (GraalVM Polyglot / JPype / JNI) | 3-6 weeks |
| **FEAT-016** | **Model Registry** (versioning, lineage, A/B testing) | 4-6 weeks |
| **FEAT-017** | **AutoML Pipeline** (architecture search + HPO) | 8-12 weeks |
| **FEAT-018** | **Streaming/Online Learning** support | 4-6 weeks |

---

## 7. Development Guidelines

### 7.1 Code Style

| Rule | Standard |
|------|----------|
| **Formatter** | Google Java Format (enforced via Spotless/Checkstyle) |
| **Line Length** | 120 chars |
| **Indentation** | 4 spaces (no tabs) |
| **Imports** | Explicit (no wildcards), sorted |
| **Braces** | K&R style (opening brace on same line) |
| **Naming** | CamelCase for classes/methods, UPPER_SNAKE for constants |
| **Records** | Use for immutable data carriers (Java 17+) |
| **Nullability** | `@Nullable`/`@NonNull` annotations (JSpecify/SpotBugs) |
| **Documentation** | Javadoc for all public APIs; inline for complex logic |

### 7.2 Testing Strategy

| Level | Scope | Tools | Target Coverage |
|-------|-------|-------|-----------------|
| **Unit** | Single class, mocked deps | JUnit 5, Mockito, AssertJ | >90% |
| **Integration** | Module interactions, real deps | JUnit 5, Testcontainers (if DB) | >70% |
| **Contract** | Config serialization, model I/O | JUnit 5, JSONAssert | 100% |
| **Property-based** | GA operators, math utils | jqwik | Key algorithms |

**Test Naming**: `ClassNameTest` for unit, `ClassNameIT` for integration
**Run Commands**:
```bash
# All tests
mvn test

# Module-specific
mvn test -pl neural-core
mvn test -pl ga-core

# Integration tests (failsafe)
mvn verify -Pintegration

# Single test
mvn test -Dtest=NeuronTest -pl neural-core
```

### 7.3 Git Workflow

| Branch | Purpose | Protection |
|--------|---------|------------|
| `main` | Production-ready, tagged releases | Required reviews (1), CI passing |
| `develop` | Integration branch for features | CI passing |
| `feature/*` | New features | PR to `develop` |
| `bugfix/*` | Bug fixes | PR to `develop` or `main` (hotfix) |
| `release/*` | Release preparation | PR to `main` |

**Commit Convention**: Conventional Commits
```
feat: add SBX crossover operator
fix: resolve output layer size mismatch in cascade GA
docs: update SETUP.md with Windows instructions
refactor: extract MatrixOps utility class
test: add integration test for data pipeline
chore: update SpotBugs to 4.8.5
```

**Release Process**:
1. Create `release/vX.Y.Z` branch from `develop`
2. Update version in parent POM (`mvn versions:set`)
3. Update `CHANGELOG.md`
4. PR to `main` → merge → tag `vX.Y.Z`
5. GitHub Actions builds and publishes artifacts

### 7.4 Code Review Checklist
- [ ] Tests added/updated for changes
- [ ] No circular dependencies introduced
- [ ] Config externalized (no new hardcoded values)
- [ ] Thread safety considered
- [ ] Logging appropriate (not too verbose, not too sparse)
- [ ] Error handling with meaningful messages
- [ ] Documentation updated (README, Javadoc, CHANGELOG)

---

## 8. Deployment

### 8.1 Build Commands

```bash
# Clean build all modules (skip tests for speed)
mvn clean install -DskipTests

# Full build with tests
mvn clean install

# Build specific module
mvn clean install -pl neural-core -am

# Build with native image profile (GUI)
mvn clean package -Pnative -pl gui-javafx
```

### 8.2 Artifacts Produced

| Module | Artifact | Type |
|--------|----------|------|
| `cli` | `cli-1.0.0-SNAPSHOT.jar` | Fat JAR (shaded, executable) |
| `gui-javafx` | `gui-javafx-1.0.0-SNAPSHOT.jar` | Modular JAR (run via javafx:run or jpackage) |
| `neural-core` | `neural-core-1.0.0-SNAPSHOT.jar` | Library JAR |
| `ga-core` | `ga-core-1.0.0-SNAPSHOT.jar` | Library JAR |
| `all` | `neural-ga-refactored-1.0.0-SNAPSHOT.pom` | Parent POM |

### 8.3 Running Applications

#### CLI (Fat JAR)
```bash
# Train
java -jar cli/target/cli-1.0.0-SNAPSHOT.jar train \
  --data data/sample/cpu.txt \
  --config config/training.yaml \
  --output models/best-model.json

# Predict
java -jar cli/target/cli-1.0.0-SNAPSHOT.jar predict \
  --model models/best-model.json \
  --data data/sample/new-data.txt \
  --output predictions.csv

# Evaluate
java -jar cli/target/cli-1.0.0-SNAPSHOT.jar evaluate \
  --model models/best-model.json \
  --data data/sample/cpu.txt

# Generate synthetic data
java -jar cli/target/cli-1.0.0-SNAPSHOT.jar generate-data \
  --type SINE_WITH_NOISE --length 1000 --noise 0.05 \
  --output data/sample/synthetic.csv
```

#### GUI (Maven)
```bash
# Development mode (hot reload via javafx:run)
mvn -pl gui-javafx javafx:run

# Production JAR (requires JavaFX modules on module path)
java --module-path ~/.m2/repository/org/openjfx \
  --add-modules javafx.controls,javafx.fxml \
  -jar gui-javafx/target/gui-javafx-1.0.0-SNAPSHOT.jar
```

#### Native Image (GraalVM)
```bash
# Requires GraalVM 17+ with native-image installed
mvn -Pnative -pl gui-javafx native:compile
# Output: gui-javafx/target/gui-javafx (native executable)
```

### 8.4 Docker Deployment

```dockerfile
# Multi-stage build
FROM maven:3.9-eclipse-temurin-17 AS builder
WORKDIR /app
COPY . .
RUN mvn clean package -DskipTests

FROM eclipse-temurin:17-jre-alpine
WORKDIR /app
COPY --from=builder /app/cli/target/cli-1.0.0-SNAPSHOT.jar app.jar
COPY --from=builder /app/config/training.yaml config/training.yaml
COPY --from=builder /app/data/sample/cpu.txt data/sample/cpu.txt
ENTRYPOINT ["java", "-jar", "app.jar"]
```

```bash
# Build
docker build -t neural-ga:latest .

# Run training
docker run -v $(pwd)/models:/app/models neural-ga:latest \
  train --data data/sample/cpu.txt --config config/training.yaml --output models/model.json
```

### 8.5 CI/CD Pipeline (GitHub Actions)

```yaml
# .github/workflows/ci.yml (already configured)
# Triggers: push to main/develop, PR to main
# Jobs:
#   - build-test: matrix (ubuntu-latest), JDK 17, mvn verify
#   - code-quality: checkstyle, spotbugs
#   - native-image: on main branch, build native GUI
#   - release: on tag push, create GitHub Release with artifacts
```

### 8.6 Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `JAVA_HOME` | JDK 17 location | Required |
| `MAVEN_OPTS` | Maven JVM options | `-Xmx2g` |
| `NEURAL_GA_CONFIG` | Config file path override | `config/training.yaml` |
| `NEURAL_GA_MODELS` | Model output directory | `models/` |
| `NEURAL_GA_DATA` | Data input directory | `data/sample/` |

---

## Appendix: Original vs Refactored Comparison

| Aspect | Original (2011) | Refactored (2024) |
|--------|-----------------|-------------------|
| **Lines of Code** | ~2,000 (2 packages) | ~15,000 (8 modules) |
| **Classes** | ~15 | ~80+ |
| **Build** | BlueJ manual compile | Maven multi-module |
| **Config** | Hardcoded in Java | YAML (type-safe) |
| **GA Operators** | Stubs only | SBX, Gaussian, Tournament, etc. |
| **Data Split** | None | Chronological 70/20/10 |
| **Normalization** | Manual | 4 methods + auto-detect |
| **Synthetic Data** | None | 7 generators |
| **Serialization** | None | JSON (full model) |
| **Tests** | 0 | 170+ |
| **GUI** | Swing | JavaFX + FXML + CSS |
| **Logging** | println | SLF4J/Logback |
| **CI/CD** | None | GitHub Actions |
| **Packaging** | .class files | Fat JAR, Native Image |

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2024-01-15 | Refactoring Team | Initial version based on refactored codebase |

---

*This document serves as the authoritative reference for the Neural-GA Refactored project. Update it as the project evolves.*