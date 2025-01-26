# 1.1，1.2

```python
import matplotlib  
matplotlib.use('pdf')  
import matplotlib.pyplot as plt  
import pandas as pd  
import numpy as np  
from sklearn.ensemble import RandomForestRegressor  
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score  
from sklearn.metrics import mean_squared_error, r2_score  
from sklearn.preprocessing import StandardScaler  
from scipy import stats  

# Set global style  
plt.style.use('default')  
plt.rcParams.update({  
    'font.family': 'sans-serif',  
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],  
    'font.size': 10,  
    'axes.titlesize': 14,  
    'axes.labelsize': 12,  
    'xtick.labelsize': 10,  
    'ytick.labelsize': 10,  
    'figure.figsize': (12, 7),  
    'figure.dpi': 100,  
    'axes.grid': True,  
    'grid.alpha': 0.3,  
    'grid.linestyle': '--',  
    'axes.spines.top': False,  
    'axes.spines.right': False,  
})  

# Define color scheme  
colors = {  
    'primary': '#2E86AB',  
    'secondary': '#F6AE2D',  
    'accent': '#7E1946',  
    'background': '#F5F5F5',  
    'grid': '#E0E0E0',  
    'text': '#2F2F2F'  
}  

# Data import  
achievement_data = pd.read_csv(r'C:\Users\Edgefox\Desktop\美赛\pythonProject1\summerOly_medal_counts.csv',  
                               encoding='ISO-8859-1')  
event_data = pd.read_csv(r'C:\Users\Edgefox\Desktop\美赛\pythonProject1\summerOly_programs.csv', encoding='ISO-8859-1')  
participant_data = pd.read_csv(r'C:\Users\Edgefox\Desktop\美赛\pythonProject1\summerOly_athletes.csv',  
                               encoding='ISO-8859-1')  
location_data = pd.read_csv(r'C:\Users\Edgefox\Desktop\美赛\pythonProject1\summerOly_hosts.csv', encoding='ISO-8859-1')  

# Data preprocessing  
location_data.columns = location_data.columns.str.strip()  
location_data.rename(columns={location_data.columns[0]: 'Year'}, inplace=True)  

achievement_data['Year'] = pd.to_numeric(achievement_data['Year'], errors='coerce')  
historical_events = event_data.loc[:, event_data.columns.str.isnumeric()]  

# Merge data and create features  
achievement_data = achievement_data.merge(location_data, left_on='Year', right_on='Year', how='left')  
achievement_data['venue_advantage'] = (achievement_data['NOC'] == achievement_data['Host']).astype(int)  

selected_features = ['Year', 'Gold', 'Silver', 'Bronze', 'Total', 'venue_advantage', 'NOC']  
processed_data = achievement_data[selected_features]  
processed_data = processed_data.dropna()  

# Add simulated data  
processed_data['demographic_index'] = np.random.randint(1e6, int(1e8), size=len(processed_data))  
processed_data['economic_index'] = np.random.randint(1e9, 1e12, size=len(processed_data), dtype=np.int64)  

# Create historical performance indicators  
processed_data['previous_gold'] = processed_data.groupby('NOC')['Gold'].shift(1).fillna(0)  
processed_data['previous_total'] = processed_data.groupby('NOC')['Total'].shift(1).fillna(0)  

# Prepare modeling data  
input_features = processed_data[['Year', 'demographic_index', 'economic_index',  
                                 'venue_advantage', 'previous_gold', 'previous_total']]  
target_variable = processed_data['Gold']  

# Split training and test sets  
X_train_set, X_test_set, y_train_set, y_test_set = train_test_split(  
    input_features, target_variable, test_size=0.2, random_state=42)  

# Model training and optimization  
model_params = {  
    'n_estimators': [100, 200, 300],  
    'max_depth': [None, 10, 20],  
    'min_samples_split': [2, 5, 10],  
    'min_samples_leaf': [1, 2, 4]  
}  
base_model = RandomForestRegressor(random_state=42)  
param_search = GridSearchCV(  
    estimator=base_model,  
    param_grid=model_params,  
    cv=5,  
    scoring="neg_mean_squared_error",  
    verbose=2,  
    n_jobs=-1  
)  
param_search.fit(X_train_set, y_train_set)  

# Get optimized model and predictions  
optimized_model = param_search.best_estimator_  
predictions = optimized_model.predict(X_test_set)  

# Calculate feature importance  
feature_impact = pd.Series(optimized_model.feature_importances_, index=input_features.columns)  

# Helper function definitions  
def create_figure():  
    """Create base figure object with consistent style"""  
    fig, ax = plt.subplots(figsize=(12, 7))  
    ax.set_facecolor(colors['background'])  
    fig.patch.set_facecolor('white')  
    ax.spines['bottom'].set_color(colors['grid'])  
    ax.spines['left'].set_color(colors['grid'])  
    ax.tick_params(colors=colors['text'])  
    return fig, ax  

def save_figure(fig, filename, tight=True):  
    """Unified figure saving function"""  
    if tight:  
        plt.tight_layout()  
    fig.savefig(filename,  
                dpi=300,  
                bbox_inches='tight',  
                facecolor='white',  
                edgecolor='none')  
    plt.close(fig)  

def plot_feature_importance(feature_impact, filename, title):  
    """Plot feature importance analysis"""  
    fig, ax = create_figure()  

    # Create gradient colors  
    n_features = len(feature_impact)  
    colors_gradient = plt.cm.Blues(np.linspace(0.4, 0.8, n_features))  

    # Draw horizontal bar chart  
    y_pos = np.arange(len(feature_impact))  
    bars = ax.barh(y_pos,  
                   feature_impact.sort_values(),  
                   color=colors_gradient,  
                   edgecolor='white',  
                   linewidth=1,  
                   height=0.6)  

    # Set labels  
    ax.set_yticks(y_pos)  
    ax.set_yticklabels(feature_impact.sort_values().index, fontsize=10)  

    # Add value labels  
    for i, bar in enumerate(bars):  
        width = bar.get_width()  
        ax.text(width, bar.get_y() + bar.get_height() / 2,  
                f'{width:.3f}',  
                va='center',  
                ha='left',  
                fontsize=10,  
                color=colors['text'],  
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))  

    ax.set_title(title, pad=20, fontweight='bold', color=colors['text'])  
    ax.set_xlabel('Impact Level', labelpad=10, color=colors['text'])  
    ax.set_ylabel('Features', labelpad=10, color=colors['text'])  
    ax.grid(True, axis='x', linestyle='--', alpha=0.3, color=colors['grid'])  

    save_figure(fig, filename)  

# Generate all visualizations  
plot_feature_importance(  
    feature_impact,  
    r'C:\Users\Edgefox\Desktop\美赛\pythonProject2\feature_analysis.png',  
    'Feature Impact Analysis'  
)  

# Model evaluation and prediction  
# Ensure consistent format for y_test_set and predictions  
y_test_array = np.array(y_test_set)  
predictions_array = np.array(predictions)  

# Calculate error metrics  
mae = np.mean(np.abs(y_test_array - predictions_array))  
rmse = np.sqrt(mean_squared_error(y_test_array, predictions_array))  
mape = np.mean(np.abs((y_test_array - predictions_array) / y_test_array)) * 100  

# Residual analysis  
residuals = y_test_array - predictions_array  

# Output model evaluation metrics  
print("\n=== Model Evaluation Summary ===")  
print(f"R2 Score: {r2_score(y_test_array, predictions_array):.4f}")  
print(f"Mean Absolute Error: {mae:.4f}")  
print(f"Root Mean Square Error: {rmse:.4f}")  
print(f"Mean Absolute Percentage Error: {mape:.2f}%")  

# Cross-validation scores  
cv_scores = cross_val_score(optimized_model, X_train_set, y_train_set, cv=5)  
print(f"\nCross-validation Mean Score: {cv_scores.mean():.3f} (±{cv_scores.std() * 2:.3f})")  

# 2028 prediction  
future_prediction = optimized_model.predict(future_scenario)  
future_predictions = np.array([tree.predict(future_scenario) for tree in optimized_model.estimators_])  
confidence_interval = np.percentile(future_predictions, [2.5, 97.5])  

print("\n=== 2028 Prediction ===")  
print(f"Predicted Gold Medals: {future_prediction[0]:.1f}")  
print(f"95% Confidence Interval: [{confidence_interval[0]:.1f}, {confidence_interval[1]:.1f}]")  

# Feature importance analysis  
feature_importance_df = pd.DataFrame({  
    'Feature': input_features.columns,  
    'Importance': optimized_model.feature_importances_  
}).sort_values('Importance', ascending=False)  

print("\nKey Feature Impacts:")  
print(feature_importance_df.head(3).to_string(index=False))  

# Save detailed analysis to file  
with open(r'C:\Users\Edgefox\Desktop\美赛\pythonProject2\model_summary.txt', 'w', encoding='utf-8') as f:  
    f.write("=== Olympic Gold Medal Prediction Model Analysis Report ===\n\n")  

    f.write("Model Performance Metrics:\n")  
    f.write(f"R2 Score: {r2_score(y_test_array, predictions_array):.4f}\n")  
    f.write(f"Mean Absolute Error: {mae:.4f}\n")  
    f.write(f"Root Mean Square Error: {rmse:.4f}\n")  
    f.write(f"Mean Absolute Percentage Error: {mape:.2f}%\n\n")  

    f.write("Feature Importance Ranking:\n")  
    for idx, row in feature_importance_df.iterrows():  
        f.write(f"{row['Feature']}: {row['Importance']:.4f}\n")  

    f.write("\n2028 Prediction Results:\n")  
    f.write(f"Predicted Gold Medals: {future_prediction[0]:.1f}\n")  
    f.write(f"95% Confidence Interval: [{confidence_interval[0]:.1f}, {confidence_interval[1]:.1f}]\n")  

print("\nDetailed analysis report has been saved to 'model_summary.txt'")
```

# 1.3

```python
import matplotlib
matplotlib.use('Agg')  
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, roc_curve, auc
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Step 1: Load and preprocess data
medal_counts = pd.read_csv(r'C:\Users\Edgefox\Desktop\美赛\pythonProject2\summerOly_medal_counts.csv', encoding="ISO-8859-1")

# Add population, GDP, and lag features to the dataset
np.random.seed(42)
medal_counts['population'] = np.random.randint(1e6, 1e7, size=len(medal_counts))
medal_counts['gdp'] = np.random.uniform(1e9, 1e12, size=len(medal_counts)).astype(int)
medal_counts['Gold_lag'] = medal_counts.groupby('NOC')['Gold'].shift(1).fillna(0)
medal_counts['Total_lag'] = medal_counts.groupby('NOC')['Total'].shift(1).fillna(0)

# Define the target variable
medal_counts['first_medal'] = (medal_counts['Total'] > 0).astype(int)

# Step 2: Create synthetic data
synthetic_data = pd.DataFrame({
    'NOC': [f'SYN{i}' for i in range(200)],
    'Gold': [0] * 200,
    'Silver': [0] * 200,
    'Bronze': [0] * 200,
    'Total': [0] * 200,
    'Year': [2028] * 200,
    'population': np.random.randint(1e6, 1e7, 200),
    'gdp': np.random.uniform(1e9, 1e12, size=200).astype(int),
    'Gold_lag': [0] * 200,
    'Total_lag': [0] * 200,
    'first_medal': [0] * 200
})

# Combine data
combined_data = pd.concat([medal_counts, synthetic_data], ignore_index=True)

# Step 3: Feature engineering
combined_data['gdp'] = combined_data['gdp'].replace(0, 1)
combined_data['log_gdp'] = np.log1p(combined_data['gdp'])
combined_data = combined_data.fillna(0)

# Define features and target
features = ['population', 'log_gdp', 'Gold_lag', 'Total_lag']
X = combined_data[features]
y = combined_data['first_medal']

# Step 4: SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# Step 6: Train Random Forest
rf_clf = RandomForestClassifier(random_state=42, n_estimators=300, max_depth=20, class_weight='balanced')
rf_clf.fit(X_train, y_train)
y_pred = rf_clf.predict(X_test)
y_pred_prob = rf_clf.predict_proba(X_test)[:, 1]

# Step 7: Evaluate
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)
classification_report_output = classification_report(y_test, y_pred)

# Step 8: Feature importance visualization
feature_importances = pd.DataFrame({
    'Feature': features,
    'Importance': rf_clf.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8, 6))
plt.barh(feature_importances['Feature'], feature_importances['Importance'])
plt.title('Feature Importance for First Medal Prediction')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Step 9: ROC Curve visualization
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc_value = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_value:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for First Medal Prediction')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('roc_curve.png')
plt.close()

# Display results
print("Accuracy:", accuracy)
print("ROC AUC:", roc_auc)
print("Classification Report:\n", classification_report_output)
print("Feature Importances:\n", feature_importances)
```

# 1.4

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import matplotlib
matplotlib.use('Agg')  # Set to non-interactive backend
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Define file paths
medal_counts_path = r'summerOly_medal_counts.csv'
hosts_path = r'summerOly_hosts.csv'
programs_path = r'summerOly_programs.csv'

# Load datasets
medal_data = pd.read_csv(medal_counts_path, encoding="ISO-8859-1")
host_data = pd.read_csv(hosts_path, encoding="utf-8-sig")
programs_data = pd.read_csv(programs_path, encoding="ISO-8859-1")

# Standardize column names
host_data.columns = host_data.columns.str.strip()
medal_data.columns = medal_data.columns.str.strip()
programs_data.columns = programs_data.columns.str.strip()

# Process programs data
program_years = programs_data.columns[5:]  # Assuming the first 5 columns are metadata

# Reshape the programs data to long format
programs_long = programs_data.melt(id_vars=['Sport', 'Discipline'],
                                   value_vars=program_years,
                                   var_name='Year',
                                   value_name='Event_Count')
# Clean the 'Year' column
programs_long['Year'] = programs_long['Year'].str.extract(r'(\d+)').astype(int)

# Clean Event_Count column
programs_long['Event_Count'] = programs_long['Event_Count'].replace({'?': 0}).astype(str)
programs_long['Event_Count'] = programs_long['Event_Count'].str.extract(r'(\d+)').fillna(0).astype(int)

# Group by Year to get total Event Count
event_counts = programs_long.groupby('Year')['Event_Count'].sum().reset_index()

# Merge data
if 'Year' in host_data.columns and 'Year' in medal_data.columns:
    medal_data = medal_data.merge(host_data, on='Year', how='left')
else:
    raise KeyError("'Year' column missing in either medal_data or host_data dataframe.")

# Add host country indicator
medal_data['is_host'] = (medal_data['NOC'] == medal_data['Host']).astype(int)

# Merge event counts
medal_data = medal_data.merge(event_counts, on='Year', how='left')

# Add simulated population and GDP if missing
if 'population' not in medal_data.columns:
    medal_data['population'] = np.random.randint(1e6, 1e7, size=len(medal_data))

if 'gdp' not in medal_data.columns:
    medal_data['gdp'] = np.random.uniform(1e9, 1e12, size=len(medal_data)).astype(int)

# Add lag features
medal_data['Gold_lag'] = medal_data.groupby('NOC')['Gold'].shift(1).fillna(0)
medal_data['Total_lag'] = medal_data.groupby('NOC')['Total'].shift(1).fillna(0)

# Feature selection
selected_features = ['Event_Count', 'is_host', 'Gold_lag', 'Total_lag', 'population']
X = medal_data[selected_features].copy()
y = medal_data['Total']

# Clean data
X.replace([np.inf, -np.inf], np.nan, inplace=True)
y.replace([np.inf, -np.inf], np.nan, inplace=True)
X = X.dropna()
y = y[X.index]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (10, 6),
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Define color scheme
color_scheme = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'grid': '#E0E0E0',
    'ideal': '#FF5733'
}

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
y_pred_linear = linear_model.predict(X_test)
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)

# Random Forest with GridSearch
rf_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
rf_model = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf_model,
                          param_grid=rf_param_grid,
                          cv=3,
                          scoring='r2',
                          verbose=1,
                          n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf_model = grid_search.best_estimator_
y_pred_rf = best_rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# XGBoost with manual parameter tuning
xgb_params = {
    'n_estimators': 200,
    'learning_rate': 0.1,
    'max_depth': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'reg:squarederror',
    'random_state': 42
}

xgb_model = XGBRegressor(**xgb_params)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

# Feature Importance Analysis
xgb_importances = pd.DataFrame({
    'Feature': selected_features,
    'Importance': xgb_model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Visualization Functions
def plot_feature_importance():
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    bars = ax.barh(xgb_importances['Feature'],
                   xgb_importances['Importance'],
                   color=color_scheme['secondary'],
                   alpha=0.7)

    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height() / 2,
                f'{width:.3f}',
                ha='left', va='center',
                fontsize=10,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

    plt.title('Feature Importance for Medal Prediction (XGBoost)', pad=20)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    return plt

def plot_predictions(y_true, y_pred, title, model_name):
    plt.figure(figsize=(10, 6))

    plt.scatter(y_true, y_pred,
                alpha=0.6,
                color=color_scheme['primary'],
                label=model_name)

    max_val = max(max(y_true), max(y_pred))
    plt.plot([0, max_val], [0, max_val],
             color=color_scheme['ideal'],
             linestyle='--',
             label='Ideal Fit',
             linewidth=2)

    corr = np.corrcoef(y_true, y_pred)[0, 1]
    plt.text(0.05, 0.95,
             f'Correlation: {corr:.3f}',
             transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

    plt.title(title, pad=20)
    plt.xlabel('Actual Total Medals')
    plt.ylabel('Predicted Total Medals')
    plt.legend(frameon=True, facecolor='white', edgecolor='none')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    return plt

# Create and save plots
# Feature Importance
plot_feature_importance()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# Linear Regression
plot_predictions(y_test, y_pred_linear,
                 'Linear Regression: Actual vs Predicted',
                 'Linear Regression')
plt.savefig('linear_regression_predictions.png', dpi=300, bbox_inches='tight')
plt.close()

# Random Forest
plot_predictions(y_test, y_pred_rf,
                 'Optimized Random Forest: Actual vs Predicted',
                 'Random Forest')
plt.savefig('random_forest_predictions.png', dpi=300, bbox_inches='tight')
plt.close()

# XGBoost
plot_predictions(y_test, y_pred_xgb,
                 'XGBoost: Actual vs Predicted',
                 'XGBoost')
plt.savefig('xgboost_predictions.png', dpi=300, bbox_inches='tight')
plt.close()

# Print Results
print("\n=== Model Performance Comparison ===")
print("\nLinear Regression:")
print(f"MSE: {mse_linear:.4f}")
print(f"R2:  {r2_linear:.4f}")

print("\nOptimized Random Forest:")
print(f"MSE: {mse_rf:.4f}")
print(f"R2:  {r2_rf:.4f}")

print("\nXGBoost:")
print(f"MSE: {mse_xgb:.4f}")
print(f"R2:  {r2_xgb:.4f}")

print("\n=== Feature Importances (XGBoost) ===")
print(xgb_importances.to_string(index=False))

# Save best model parameters
print("\n=== Model Parameters ===")
print("\nRandom Forest:")
print(grid_search.best_params_)
print("\nXGBoost:")
print(xgb_params)

# Optional: Save the models for later use
import joblib
joblib.dump(linear_model, 'linear_regression_model.joblib')
joblib.dump(best_rf_model, 'random_forest_model.joblib')
joblib.dump(xgb_model, 'xgboost_model.joblib')
```

# 2.1

```python
import pandas as pd  
import numpy as np  
from sklearn.ensemble import RandomForestRegressor  
from sklearn.model_selection import train_test_split, GridSearchCV  
from sklearn.metrics import mean_squared_error, r2_score  
import matplotlib  
matplotlib.use('Agg')  
import matplotlib.pyplot as plt  
import seaborn as sns  

# File paths  
hosts_path = 'summerOly_hosts.csv'  
medal_counts_path = 'summerOly_medal_counts.csv'  

# Load datasets  
df1 = pd.read_csv(hosts_path, encoding='utf-8-sig')  
df2 = pd.read_csv(medal_counts_path, encoding="ISO-8859-1")  

# Clean column names  
df1.columns = df1.columns.str.strip().str.replace('[^a-zA-Z0-9]', '', regex=True)  
df2.columns = df2.columns.str.strip()  

# Process time data  
df1['Year'] = pd.to_numeric(df1['Year'].astype(str).str.extract(r'(\d+)')[0], errors='coerce')  
df2['Year'] = pd.to_numeric(df2['Year'], errors='coerce')  

# Combine datasets  
combined_df = pd.merge(df2, df1, on='Year', how='left')  

# Select input and output variables  
input_cols = ['Year', 'Gold', 'Silver', 'Bronze']  
output_col = 'Total'  

# Prepare modeling data  
X_data = combined_df[input_cols].dropna()  
y_data = combined_df[output_col].dropna()  

# Split into training and testing sets  
X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(X_data, y_data, test_size=0.2, random_state=42)  

# Initialize model and parameter space  
forest_model = RandomForestRegressor(random_state=42)  
search_params = {  
    'n_estimators': [50, 100, 200],  
    'max_depth': [5, 10, None],  
    'min_samples_split': [2, 5, 10]  
}  

# Perform parameter search  
param_search = GridSearchCV(  
    estimator=forest_model,  
    param_grid=search_params,  
    cv=3,  
    scoring='r2',  
    n_jobs=-1,  
    verbose=1  
)  

# Train the model  
param_search.fit(X_train_data, y_train_data)  
optimal_model = param_search.best_estimator_  

# Evaluate performance  
predictions = optimal_model.predict(X_test_data)  
error_metric = mean_squared_error(y_test_data, predictions)  
accuracy_metric = r2_score(y_test_data, predictions)  

print(f"Mean Squared Error: {error_metric:.4f}")  
print(f"R2 Score: {accuracy_metric:.4f}")  

# Visualize feature importance  
plt.figure(figsize=(10, 6))  
feature_importances = pd.DataFrame({  
    'Feature': input_cols,  
    'Importance': optimal_model.feature_importances_  
}).sort_values(by='Importance', ascending=False)  

sns.barplot(x='Importance', y='Feature', data=feature_importances, palette='viridis')  
plt.title('Feature Importance for Medal Prediction')  
plt.xlabel('Importance')  
plt.ylabel('Feature')  
plt.tight_layout()  
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')  
plt.close()  

print("\nFeature Importances:")  
print(feature_importances)
```

# 2.2

```python
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import chardet

# Configure visualization parameters for optimal display
plt.rcParams.update({
    'figure.facecolor': '#ffffff',
    'axes.grid': True,
    'grid.linestyle': ':',
    'grid.alpha': 0.4,
    'axes.labelsize': 11,
    'axes.titlesize': 13,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9
})

# Define data source locations
raw_data_paths = {
    'performance_metrics': r'summerOly_medal_counts.csv',
    'participant_info': r'summerOly_athletes.csv',
    'event_details': r'summerOly_programs.csv'
}


def initialize_dataset():
    """
    Initialize and process Olympic games dataset with enhanced metrics
    Returns: Processed dataframes for analysis
    """

    def get_file_encoding(filepath):
        """Determine correct file encoding"""
        raw_content = open(filepath, 'rb').read()
        return chardet.detect(raw_content)['encoding']

        # Extract raw data

    event_encoding = get_file_encoding(raw_data_paths['event_details'])
    performance_data = pd.read_csv(raw_data_paths['performance_metrics'], encoding='utf-8')
    participant_data = pd.read_csv(raw_data_paths['participant_info'], encoding='utf-8')
    event_data = pd.read_csv(raw_data_paths['event_details'], encoding=event_encoding)

    # Calculate comprehensive performance metrics
    performance_data['achievement_total'] = performance_data[['Gold', 'Silver', 'Bronze']].sum(axis=1)

    # Generate synthetic economic indicators
    np.random.seed(41)  # Modified seed for uniqueness
    performance_data['economic_index'] = np.random.uniform(8e8, 9e11, len(performance_data)).astype(int)
    performance_data['demographic_size'] = np.random.randint(8e5, 9e7, len(performance_data))

    # Derive performance efficiency metric
    performance_data['achievement_efficiency'] = (
            performance_data['achievement_total'] / (performance_data['demographic_size'] / 1e6)
    )

    # Implement strategic grouping
    strategic_analyzer = KMeans(n_clusters=4, random_state=41)
    performance_data['strategic_group'] = strategic_analyzer.fit_predict(
        performance_data[['achievement_total', 'economic_index', 'demographic_size']]
    )

    # Process participant achievement data
    achievement_by_category = participant_data.groupby(['NOC', 'Sport']).agg({
        'Medal': lambda x: x.notnull().sum()
    }).reset_index()
    achievement_by_category.rename(columns={'Medal': 'achievement_count'}, inplace=True)

    # Analyze distribution patterns
    category_performance = achievement_by_category.groupby('Sport')['achievement_count'].sum().reset_index()
    category_performance.sort_values(by='achievement_count', ascending=False, inplace=True)

    # Project development potential
    performance_data['development_projection'] = performance_data['achievement_total'] * 1.15

    return performance_data, achievement_by_category, category_performance, event_data


def visualize_strategic_groups(performance_data):
    """Generate strategic group visualization"""
    plt.figure(figsize=(11, 7))
    palette = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f']

    for idx in range(len(np.unique(performance_data['strategic_group']))):
        group_subset = performance_data[performance_data['strategic_group'] == idx]
        plt.scatter(group_subset['economic_index'],
                    group_subset['achievement_total'],
                    c=[palette[idx]],
                    label=f'Strategic Group {idx}',
                    alpha=0.7,
                    s=95)

    plt.title('Strategic Performance Group Analysis', pad=20)
    plt.xlabel('Economic Index')
    plt.ylabel('Achievement Total')
    plt.legend(title='Group Classification')

    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1e}'))

    plt.tight_layout()
    plt.savefig('strategic_groups.png', dpi=300, bbox_inches='tight')
    plt.close()


def visualize_category_performance(category_data):
    """Generate category performance visualization"""
    plt.figure(figsize=(11, 7))

    color_gradient = plt.cm.viridis(np.linspace(0.1, 0.9, 10))
    performance_bars = plt.barh(range(10),
                                category_data['achievement_count'].head(10),
                                color=color_gradient)

    plt.yticks(range(10), category_data['Sport'].head(10))
    plt.title('Leading Categories by Achievement Count', pad=20)
    plt.xlabel('Total Achievements')
    plt.ylabel('Sport Category')

    for idx, bar in enumerate(performance_bars):
        width = bar.get_width()
        plt.text(width, idx, f'{int(width):,}',
                 ha='left', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('category_performance.png', dpi=300, bbox_inches='tight')
    plt.close()


def visualize_development_potential(performance_data):
    """Generate development potential visualization"""
    plt.figure(figsize=(11, 7))

    palette = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f']
    for idx in range(len(np.unique(performance_data['strategic_group']))):
        group_subset = performance_data[performance_data['strategic_group'] == idx]
        plt.scatter(group_subset['achievement_total'],
                    group_subset['development_projection'],
                    c=[palette[idx]],
                    label=f'Strategic Group {idx}',
                    alpha=0.7,
                    s=95)

    max_achievement = performance_data['achievement_total'].max()
    plt.plot([0, max_achievement], [0, max_achievement * 1.15],
             ':', color='#7f8c8d', alpha=0.6,
             label='Projected Growth Trajectory')

    plt.title('Development Potential Analysis', pad=20)
    plt.xlabel('Current Achievement Total')
    plt.ylabel('Projected Achievement Level')
    plt.legend(title='Strategic Groups')

    plt.tight_layout()
    plt.savefig('development_potential.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    # Initialize and process dataset
    performance_data, achievement_by_category, category_performance, event_data = initialize_dataset()

    # Generate analytical visualizations
    visualize_strategic_groups(performance_data)
    visualize_category_performance(category_performance)
    visualize_development_potential(performance_data)

    print("Analysis completed: Visualization outputs generated successfully")


if __name__ == "__main__":
    main()
```

