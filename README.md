# Data Engineering at Scale Project IISC - Team Data Warriors

 
## Predicting EV Adoption vs ICE Vehicles: A Machine Learning Approach - Project Walkthrough

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Data Flow](#data-flow)
4. [Code Structure](#code-structure)
5. [Component Breakdown](#component-breakdown)
6. [How to Run](#how-to-run)
7. [Understanding the Model](#understanding-the-model)

---

## Project Overview

This project predicts **EV (Electric Vehicle) adoption probability** for 333 Indian cities over 120 months (2015-2024) using PySpark and Machine Learning.

**Key Technologies:**
- **HDFS**: Distributed file storage
- **PySpark**: Large-scale data processing
- **MLlib**: Machine learning algorithms (Random Forest)
- **Python**: Data analysis and modeling

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        PROJECT ARCHITECTURE                      │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐
│   CSV Dataset    │  (333 cities × 120 months = 39,960 rows)
│  9.6 MB on disk  │
└────────┬─────────┘
         │
         │ Upload
         ▼
┌──────────────────────────────────────────────────────────┐
│                    HDFS (Hadoop)                         │
│  ┌────────────────┐          ┌──────────────────┐       │
│  │   NameNode     │◄────────►│   DataNode       │       │
│  │  (Metadata)    │          │  (Data Blocks)   │       │
│  │  Port: 9870    │          │  Port: 9864      │       │
│  └────────────────┘          └──────────────────┘       │
│  Storage: /des/data/                                     │
│  Quota: 30 GB                                            │
└────────────────────┬─────────────────────────────────────┘
                     │
                     │ Read via HDFS URI
                     │ (hdfs://localhost:9000/des/data/...)
                     ▼
┌──────────────────────────────────────────────────────────┐
│                  PySpark Application                     │
│                                                          │
│  ┌────────────┐    ┌────────────┐    ┌──────────────┐  │
│  │  eda.py    │    │load_data.py│    │train_model.py│  │
│  │(Analysis)  │    │ (Loader)   │    │  (ML Model)  │  │
│  └─────┬──────┘    └─────┬──────┘    └──────┬───────┘  │
│        │                 │                   │          │
│        └─────────────────┴───────────────────┘          │
│                          │                              │
│                          ▼                              │
│                 ┌─────────────────┐                     │
│                 │  Spark Session  │                     │
│                 │  (Executor)     │                     │
│                 └─────────────────┘                     │
└────────────────────┬─────────────────────────────────────┘
                     │
                     │ Write Results
                     ▼
┌──────────────────────────────────────────────────────────┐
│                    Output Directory                      │
│                                                          │
│  • predictions_all.csv       (2 MB)                      │
│  • city_summary.csv          (26 KB)                     │
│  • descriptive_stats.csv     (2.4 KB)                    │
│  • target_correlations.csv   (1.3 KB)                    │
│  • yearly_trends.csv         (320 B)                     │
│  • metrics.json              (212 B)                     │
└──────────────────────────────────────────────────────────┘
```

---

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         DATA PIPELINE                            │
└─────────────────────────────────────────────────────────────────┘

CSV File (Local)
    │
    │ hdfs dfs -put
    ▼
HDFS Storage (/des/data/)
    │
    │ spark.read.csv()
    ▼
DataFrame (39,960 rows × 35 columns)
    │
    ├──────────────────┬──────────────────┐
    │                  │                  │
    ▼                  ▼                  ▼
  EDA Path      Train-Test Split    Feature Engineering
    │              (80-20)               │
    │                  │                 │
    │                  ▼                 ▼
    │          ┌───────────────┐   ┌─────────────┐
    │          │  Train: 80%   │   │  Encoding   │
    │          │  Test:  20%   │   │  Scaling    │
    │          └───────┬───────┘   └──────┬──────┘
    │                  │                  │
    ▼                  ▼                  ▼
Descriptive      Random Forest       Feature Vector
Statistics           Model          (34 features)
    │                  │                  │
    │                  │                  │
    ▼                  ▼                  ▼
Correlations     Predictions         Evaluation
    │                  │                  │
    │                  │                  │
    └──────────────────┴──────────────────┘
                       │
                       ▼
                 Output Files
```

---

## Code Structure

```
ev_vs_ice/
│
├── ev_ice_timeseries_333_real_indian_cities_120months_with_state.csv
│   └── Raw dataset (9.6 MB)
│
├── src/
│   ├── __init__.py
│   ├── load_data.py          # Data loading utilities
│   ├── eda.py                # Exploratory Data Analysis
│   └── train_model.py        # ML model training & prediction
│
├── output/                   # Generated results
│   ├── predictions_all.csv
│   ├── city_summary.csv
│   ├── descriptive_stats.csv
│   ├── target_correlations.csv
│   ├── yearly_trends.csv
│   └── metrics.json
│
├── requirements.txt          # Python dependencies
├── README.md                 # Setup instructions
└── PROJECT_WALKTHROUGH.md    # This file
```

---

## Component Breakdown

### 1. **load_data.py** - Data Loader

**Purpose:** Initialize Spark session and load CSV from HDFS

```python
┌────────────────────────────────────────┐
│         load_data.py Flow              │
└────────────────────────────────────────┘

create_spark()
    │
    ├── Configure Spark session
    ├── Set memory (4GB driver/executor)
    ├── Set HADOOP_CONF_DIR
    └── Return SparkSession
        │
        ▼
load_dataset(spark, hdfs_path)
    │
    ├── Read CSV with schema inference
    ├── Parse dates
    └── Return DataFrame
```

**Key Functions:**
- `create_spark()`: Creates Spark session with HDFS configuration
- `load_dataset()`: Reads CSV from HDFS into DataFrame

**Example Usage:**
```python
spark = create_spark()
df = load_dataset(spark, "hdfs://localhost:9000/des/data/file.csv")
df.show(5)  # Display first 5 rows
```

---

### 2. **eda.py** - Exploratory Data Analysis

**Purpose:** Analyze dataset characteristics, correlations, and distributions

```
┌────────────────────────────────────────────────────────────┐
│                    eda.py Workflow                         │
└────────────────────────────────────────────────────────────┘

Load Data
    │
    ▼
schema_info(df)
    │ Output: Column names, types, row count
    ▼
missing_values_analysis(df)
    │ Output: Null count per column
    ▼
descriptive_statistics(df)
    │ Output: Mean, std, min, max, count
    │ Saves: descriptive_stats.csv
    ▼
categorical_analysis(df)
    │ Output: Unique cities (333), states (34)
    │ Saves: city_state_counts.csv
    ▼
correlation_analysis(df, target)
    │ Output: Feature correlations with target
    │ Saves: target_correlations.csv
    │ Top correlations:
    │   • charging_stations_per_10k: 0.564
    │   • ev_awareness_score: 0.551
    │   • ev_subsidy_amount: 0.536
    ▼
temporal_analysis(df, target)
    │ Output: Yearly adoption trends
    │ Saves: yearly_trends.csv
    ▼
distribution_analysis(df, target)
    │ Output: Histogram, percentiles
    │ Range: 0.08 to 0.72
    │ Median: 0.35
    └──► EDA Complete
```

**Key Insights from EDA:**
1. **No missing data** - 100% complete dataset
2. **Strong predictors:**
   - Charging infrastructure (0.564 correlation)
   - Public awareness (0.551)
   - Government subsidies (0.536)
3. **Temporal trend:** Adoption grew from 28.6% (2015) to 42% (2024)

---

### 3. **train_model.py** - ML Model Training

**Purpose:** Build and train Random Forest model to predict EV adoption

```
┌────────────────────────────────────────────────────────────┐
│              train_model.py Pipeline                       │
└────────────────────────────────────────────────────────────┘

Load Data from HDFS
    │
    ▼
select_columns(df)
    │
    ├── Extract day from date
    ├── Identify categorical: ['city', 'state']
    ├── Identify numeric: 32 features
    └── Total features: 34
        │
        ▼
build_pipeline()
    │
    ├── Stage 1: StringIndexer
    │   └── Convert categorical → numeric indices
    │       • city → city_index (0-332)
    │       • state → state_index (0-33)
    │
    ├── Stage 2: OneHotEncoder
    │   └── Create binary vectors
    │       • city_index → city_vec (sparse vector)
    │       • state_index → state_vec (sparse vector)
    │
    ├── Stage 3: VectorAssembler
    │   └── Combine all features into single vector
    │       • Input: city_vec + state_vec + 32 numeric
    │       • Output: features (length 367)
    │
    └── Stage 4: RandomForestRegressor
        └── Train model
            • numTrees: 100
            • maxDepth: 10
            • Handles 34 input features
            │
            ▼
Train-Test Split (80-20)
    │
    ├── Train: 31,968 rows (80%)
    └── Test:   7,992 rows (20%)
        │
        ▼
Fit Pipeline on Training Data
    │ (Trains Random Forest on 100 trees)
    ▼
Predict on Test Data
    │
    ├── Generate predictions for all rows
    └── Save predictions_all.csv
        │
        ▼
Evaluate Performance
    │
    ├── Regression Metrics:
    │   • RMSE: 0.0093 (very low error!)
    │   • R²: 0.9916 (99.16% accuracy!)
    │   • MAE: 0.0072
    │
    └── Classification Metrics (threshold=0.5):
        • Accuracy: 98.19%
        • F1 Score: 98.09%
        │
        ▼
City-Level Summary
    │ Group by city, calculate RMSE per city
    │ Saves: city_summary.csv
    └──► Training Complete
```

---

## Feature Engineering Explained

```
┌────────────────────────────────────────────────────────────┐
│            Feature Transformation Flow                     │
└────────────────────────────────────────────────────────────┘

INPUT FEATURES (Raw):
├── city: "Mumbai" (string)
├── state: "Maharashtra" (string)
└── 32 numeric features (fuel_price, aqi, charging_stations, etc.)

    │
    │ StringIndexer
    ▼
INDEXED:
├── city_index: 0 (numeric label for "Mumbai")
├── state_index: 12 (numeric label for "Maharashtra")
└── 32 numeric features (unchanged)

    │
    │ OneHotEncoder
    ▼
ENCODED:
├── city_vec: [0,0,0,...,1,0,0] (sparse vector, length 333)
├── state_vec: [0,0,...,1,0] (sparse vector, length 34)
└── 32 numeric features (unchanged)

    │
    │ VectorAssembler
    ▼
FEATURE VECTOR:
└── features: [city_vec (333) + state_vec (34) + numeric (32)]
              = Dense vector of length 367

    │
    │ Random Forest
    ▼
PREDICTION:
└── ev_adoption_probability: 0.34 (predicted value)
```

**Why this approach?**
- **StringIndexer**: Converts categorical text → numbers (required for ML)
- **OneHotEncoder**: Prevents ordinal assumptions (Mumbai ≠ 2× Delhi)
- **VectorAssembler**: Combines all features into single input vector
- **Random Forest**: Learns non-linear relationships between features and target

---

## Model Algorithm: Random Forest Explained

```
┌────────────────────────────────────────────────────────────┐
│              Random Forest Architecture                    │
└────────────────────────────────────────────────────────────┘

Input: Feature Vector (367 features)
    │
    ├──► Tree 1 (Depth 10)
    │      ├── Split on charging_stations_per_10k
    │      ├── Split on ev_awareness_score
    │      └── ... (continues to depth 10)
    │           └──► Prediction₁: 0.35
    │
    ├──► Tree 2 (Depth 10)
    │      ├── Split on ev_subsidy_amount
    │      ├── Split on battery_range_km
    │      └── ...
    │           └──► Prediction₂: 0.33
    │
    ├──► Tree 3 ... Tree 100
    │      └──► Prediction₃...₁₀₀
    │
    └──► Average all 100 predictions
         └──► Final Prediction: 0.34

KEY PARAMETERS:
• numTrees = 100     (more trees = better accuracy, slower training)
• maxDepth = 10      (deeper trees = capture complexity, risk overfitting)
• handleInvalid = skip (ignore rows with invalid values)
```

**Advantages of Random Forest:**
1. **High Accuracy**: Ensemble of 100 trees reduces error
2. **Non-linear**: Captures complex relationships
3. **Feature Importance**: Identifies key predictors
4. **Robust**: Less prone to overfitting than single decision tree

---

## How to Run

### Step 1: Start HDFS

```bash
# Start NameNode and DataNode
hdfs --daemon start namenode
hdfs --daemon start datanode

# Verify cluster is running
jps  # Should show: NameNode, DataNode

# Check web UI
open http://localhost:9870
```

### Step 2: Upload Data to HDFS

```bash
# Create HDFS directory
hdfs dfs -mkdir -p /des/data

# Upload CSV file
hdfs dfs -put -f ev_ice_timeseries_333_real_indian_cities_120months_with_state.csv /des/data/

# Verify upload
hdfs dfs -ls -h /des/data/
```

### Step 3: Run EDA

```bash
# Activate virtual environment
source .venv/bin/activate

# Set Hadoop configuration
export HADOOP_CONF_DIR=/opt/homebrew/Cellar/hadoop/3.4.2/libexec/etc/hadoop

# Run EDA script
python src/eda.py \
  --hdfs-path hdfs://localhost:9000/des/data/ev_ice_timeseries_333_real_indian_cities_120months_with_state.csv \
  --output-dir ./output
```

**Expected Output:**
- Schema information
- Missing values analysis (0 nulls)
- Descriptive statistics
- Correlation analysis (top features identified)
- Temporal trends (2015-2024)
- Distribution histogram

### Step 4: Run Model Training

```bash
python src/train_model.py \
  --hdfs-path hdfs://localhost:9000/des/data/ev_ice_timeseries_333_real_indian_cities_120months_with_state.csv \
  --output-dir ./output
```

**Expected Output:**
```
Training with ALL features:
  Categorical columns (2): ['city', 'state']
  Numeric columns (32): [...]
  Total feature count: 34

Regression Metrics: 
  {'rmse': 0.0093, 'r2': 0.9916, 'mae': 0.0072}

Classification Metrics (threshold=0.5):
  {'accuracy': 0.9819, 'f1': 0.9809}

Files saved:
  ✓ predictions_all.csv
  ✓ city_summary.csv
  ✓ metrics.json
```

### Step 5: View Results

```bash
# View first 10 predictions
head -11 output/predictions_all.csv

# View city summary
head -11 output/city_summary.csv

# Check metrics
cat output/metrics.json
```

---

## Understanding the Model

### What does the model predict?

**Target Variable:** `ev_adoption_probability`
- Range: 0.08 to 0.72 (8% to 72%)
- Interpretation: Probability that a resident will adopt EV
- Example: 0.35 = 35% adoption probability

### How does it make predictions?

1. **Input:** City, state, and 32 numeric features (fuel price, AQI, subsidies, etc.)
2. **Process:** 100 decision trees vote on the prediction
3. **Output:** Average of all tree predictions

### What makes it accurate?

**R² = 0.9916** means:
- Model explains 99.16% of variance in EV adoption
- Only 0.84% is unexplained (noise or missing factors)

**RMSE = 0.0093** means:
- Average prediction error is less than 1%
- Very precise predictions

### Key Predictors (by importance):

```
┌────────────────────────────────────────────────────────┐
│         Top Factors Driving EV Adoption                │
└────────────────────────────────────────────────────────┘

1. Charging Infrastructure (0.564 correlation)
   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 56%
   
2. Public Awareness (0.551 correlation)
   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 55%
   
3. Government Subsidies (0.536 correlation)
   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 54%
   
4. Year (temporal trend) (0.421 correlation)
   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 42%
   
5. Battery Range (0.322 correlation)
   ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 32%
```

---

## Performance Metrics Explained

### Regression Metrics

**1. RMSE (Root Mean Squared Error) = 0.0093**
```
RMSE measures average prediction error

Lower is better!

Example:
  Actual: 0.35
  Predicted: 0.34
  Error: 0.01 (1%)
  
Average error across all predictions: 0.0093 (0.93%)
```

**2. R² (R-squared) = 0.9916**
```
R² measures how well model explains variance

Range: 0 (worst) to 1 (perfect)

0.9916 = Model explains 99.16% of variance
         Only 0.84% unexplained

Interpretation: Excellent fit!
```

**3. MAE (Mean Absolute Error) = 0.0072**
```
MAE measures average absolute error

Example:
  |Actual - Predicted| = |0.35 - 0.34| = 0.01
  
Average: 0.0072 (0.72%)
```

### Classification Metrics

**Threshold = 0.5:** Predictions ≥ 0.5 → "High Adoption", < 0.5 → "Low Adoption"

**1. Accuracy = 98.19%**
```
Accuracy = (Correct Predictions) / (Total Predictions)

Out of 7,992 test samples:
  Correctly predicted: 7,848
  Incorrectly predicted: 144
  
Accuracy = 7848 / 7992 = 98.19%
```

**2. F1 Score = 98.09%**
```
F1 = Harmonic mean of Precision and Recall

Balances false positives and false negatives

98.09% = Near-perfect balance
```

---

## Real-World Example

### Prediction for Mumbai (January 2015):

```
┌────────────────────────────────────────────────────────┐
│              Mumbai EV Adoption Prediction             │
└────────────────────────────────────────────────────────┘

INPUT FEATURES:
├── city: "Mumbai"
├── state: "Maharashtra"
├── year: 2015
├── month: 1
├── fuel_price_per_litre: 85.50
├── charging_stations_per_10k: 2.5
├── ev_awareness_score: 4.2
├── ev_subsidy_amount: 45000
└── ... (28 more features)

    │ Model Processing
    ▼

RANDOM FOREST COMPUTATION:
├── Tree 1 predicts: 0.342
├── Tree 2 predicts: 0.338
├── Tree 3 predicts: 0.346
└── ... (97 more trees)

    │ Average
    ▼

FINAL PREDICTION: 0.344
ACTUAL VALUE:     0.340
ERROR:            0.004 (0.4%)

✓ Highly accurate prediction!
```

---

## Common Questions

### Q1: Why use HDFS for a 9.6 MB file?

**A:** This project demonstrates **scalability**. HDFS handles:
- Terabytes of data across multiple nodes
- Fault tolerance (data replication)
- Distributed processing

For production with millions of cities or real-time data, HDFS is essential.

---

### Q2: Why Random Forest over Linear Regression?

**A:** EV adoption has **non-linear relationships**:
- Charging stations have diminishing returns
- Subsidies have threshold effects
- City characteristics interact in complex ways

Random Forest captures these better than linear models.

---

### Q3: How to improve model performance?

**Suggestions:**
1. **Add features:** Weather, policy changes, competitor models
2. **Hyperparameter tuning:** Grid search for optimal numTrees, maxDepth
3. **Feature engineering:** Interaction terms (subsidy × awareness)
4. **Ensemble methods:** Combine Random Forest with Gradient Boosting

---

### Q4: How to deploy this model?

**Production Pipeline:**
```
Real-time Data → HDFS → Spark Streaming → Model → API → Dashboard
```

**Steps:**
1. Save trained model: `model.save("hdfs://path/to/model")`
2. Load in production: `model = RandomForestModel.load(...)`
3. Create REST API (Flask/FastAPI)
4. Serve predictions on demand

---

## Summary

```
┌────────────────────────────────────────────────────────┐
│                   Project Summary                      │
└────────────────────────────────────────────────────────┘

GOAL:
  Predict EV adoption probability for Indian cities

DATA:
  333 cities × 120 months = 39,960 records
  35 features (economic, infrastructure, environmental)

APPROACH:
  1. Store data in HDFS (distributed storage)
  2. Analyze with PySpark (distributed computing)
  3. Train Random Forest (ensemble ML)
  4. Evaluate with multiple metrics

RESULTS:
  ✓ 99.16% accuracy (R²)
  ✓ 0.93% average error (RMSE)
  ✓ 98.19% classification accuracy
  
KEY INSIGHTS:
  • Charging infrastructure is #1 predictor
  • Adoption grew 48% from 2015 to 2024
  • Government subsidies strongly influence adoption
  
DELIVERABLES:
  • Full predictions for all 39,960 records
  • City-level performance metrics
  • Comprehensive EDA reports
  • Reproducible ML pipeline
```

---

## Next Steps

1. **Experiment:** Try different ML algorithms (Gradient Boosting, Neural Networks)
2. **Visualize:** Create charts for trends and predictions
3. **Optimize:** Tune hyperparameters for better accuracy
4. **Scale:** Add more cities, years, or real-time data streams
5. **Deploy:** Build web app or API for live predictions

---

**Questions?** Refer to `README.md` for setup or explore the code in `src/`!
