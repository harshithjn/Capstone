# Runtime Prediction Dashboard with Model Explainability

A clean, professional dashboard that provides comprehensive model explainability for runtime predictions using four machine learning models: Linear Regression, Decision Tree, Random Forest, and XGBoost.

## Features

### Model Explainability Components

1. **Feature Contribution Analysis**
   - Shows how each feature contributes to the prediction
   - Positive values increase runtime, negative values decrease it
   - Available for all four models using SHAP-like methodology
   - Interactive charts with tabbed interface

2. **Feature Importance**
   - Displays the most important features for tree-based models
   - Shows relative importance scores
   - Helps understand which metrics matter most

3. **Decision Tree Rules**
   - Shows the actual decision path taken by the Decision Tree model
   - Displays the conditions that led to the prediction
   - Easy-to-understand IF-THEN rules

4. **Nearest Neighbors**
   - Finds the 3 most similar historical runs from training data
   - Uses Euclidean distance to measure similarity
   - Provides context by showing similar past predictions

### UI Features

- **Clean White Theme**: Professional light UI with Matter Regular font
- **Interactive Charts**: Powered by Chart.js with hover effects
- **Tabbed Interface**: Organized explainability sections
- **Responsive Design**: Works on desktop and mobile
- **Compact Sidebar**: Streamlined input form with quick data entry
- **Real-time Updates**: All explanations update when you make new predictions

## Installation & Setup

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Dashboard**
   ```bash
   python app.py
   ```

3. **Access Dashboard**
   Open http://localhost:5000 in your browser

## Usage

### Making Predictions

1. **Quick Input**: Use the compact bulk paste feature to quickly fill all fields
   ```
   Example: ML,ml_resnet,5,8,16,70,5.6,8,0.5,120,40,160,3
   ```

2. **Manual Input**: Fill each field individually with your system metrics

3. **Get Predictions**: Click "Predict Runtime" to see results and explanations

### Understanding the Results

#### Prediction Cards
Each model shows its prediction with a clear, highlighted runtime value in an elegant card layout.

#### Feature Contributions
- **Green bars**: Features that increase runtime
- **Red bars**: Features that decrease runtime  
- **Bar length**: Magnitude of the impact
- Available for all models using advanced feature analysis

#### Feature Importance
- Shows which features the tree models consider most important
- Higher values = more important for making predictions

#### Decision Rules
- Shows the exact path the Decision Tree took
- Read as: "IF condition1 AND condition2 THEN prediction"

#### Similar Runs
- Historical runs with similar input characteristics
- Helps validate if the prediction makes sense

## Technical Architecture

### Backend (Flask)
- **Advanced Feature Analysis**: Uses model coefficients and feature importance for contribution analysis
- **Feature Importance**: Extracts from model.feature_importances_
- **Decision Paths**: Uses sklearn's decision_path() method with pipeline support
- **Nearest Neighbors**: sklearn.neighbors.NearestNeighbors with Euclidean distance

### Frontend (HTML/JS)
- **Chart.js**: Interactive bar charts for feature contributions and importance
- **Matter Regular Font**: Professional typography
- **Clean White Theme**: Modern, accessible design
- **Responsive CSS**: Mobile-friendly layout

### Data Flow
```
User Input → Flask Backend → ML Models → Predictions
                ↓
         Explainability Analysis
                ↓
    Feature Contributions + Feature Importance + Decision Rules + Nearest Neighbors
                ↓
         JSON Response → Frontend → Interactive Charts
```

## Model Support

| Model | Feature Contributions | Feature Importance | Decision Rules | Nearest Neighbors |
|-------|----------------------|-------------------|----------------|-------------------|
| Linear Regression | ✅ | ❌ | ❌ | ✅ |
| Decision Tree | ✅ | ✅ | ✅ | ✅ |
| Random Forest | ✅ | ✅ | ❌ | ✅ |
| XGBoost | ✅ | ✅ | ❌ | ✅ |

## Example Output

When you input system metrics like:
- ML workload, 8 cores, 16GB RAM, 70% CPU

The dashboard shows:

**Predictions:**
- Linear Regression → 11.67 sec
- Decision Tree → 15.44 sec  
- Random Forest → 15.91 sec
- XGBoost → 16.29 sec

**Feature Contributions:**
- memory_pressure: +2.2 sec
- io_intensity: +1.6 sec
- cpu_cores: -3.4 sec

**Most Important Features:**
- memory_pressure (0.31)
- effective_cpu (0.25)
- io_intensity (0.18)

**Decision Path:**
- IF workload_complexity > 4
- AND cpu_cores < 8  
- AND memory_pressure > 0.5
- THEN runtime ≈ 15.4 sec

**Similar Runs:**
- Run 1842 → 15.2 sec
- Run 2033 → 16.1 sec
- Run 2210 → 15.8 sec

## Dependencies

- Flask 2.3.3
- pandas 2.2.3  
- scikit-learn 1.7.2
- XGBoost 3.2.0
- SHAP 0.51.0
- Chart.js (CDN)

This explainability dashboard makes ML predictions transparent and trustworthy with a clean, professional interface.