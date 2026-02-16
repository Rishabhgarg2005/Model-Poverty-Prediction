import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import KNNImputer
data=pd.read_csv("train_hh_features.csv")  # Fixed: Added ../ to go to parent directory
print(data.info())
print(data.isnull().sum()/len(data)*100)

categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    data[col].fillna(data[col].mode()[0], inplace=True)
numeric_columns = data.select_dtypes(include=[np.number]).columns
for col in numeric_columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')
    if len(numeric_columns)>0:
       imputer =KNNImputer(n_neighbors=5)
       data[numeric_columns]=imputer.fit_transform(data[numeric_columns])

print(data.isnull().sum())
def convert(feat):
   if pd.isna(feat):
     return 0
   if feat =="male":
      return 1
   else:
      return 2
   
data["male"]=data["male"].apply(convert)
def convert2(feat):
   if pd.isna(feat):
      return 0
   if feat =="owner":
      return 1
   else:
      return 2
data["owner"]=data["owner"].apply(convert2)
def convert3(feat):
   if pd.isna(feat):
      return 0
   if feat =="Access":
      return 1
   else:
      return 2
data["water"]=data["water"].apply(convert3)
data["toilet"]=data["toilet"].apply(convert3)
data["sewer"]=data["sewer"].apply(convert3)
data["elect"]=data["elect"].apply(convert3)
print(data.head())

def convert4(feat):
   if pd.isna(feat):
      return 0
   if feat =="Piped water into dwelling":
        return 1
   if feat =="Protected dug well":
       return 3
   if feat =="Surface water":
      return 4
   else:
      return 5
data["water_source"]=data["water_source"].apply(convert4)

def convert5(feat):
   if pd.isna(feat):
      return 0
   if feat =="A piped sewer system":
       return 1
   if feat =="A septic tank":
      return 2
   if feat =="Pit latrine with slab":
      return 3
   if feat =="No facilities or bush or field":
      return 4
   else:
      return 5
data["sanitation_source"]=data["sanitation_source"].apply(convert5)
def convert6(feat):
   if pd.isna(feat):
        return 0
   if feat =="Detached house":
    return 1
   
   if feat =="Several buildings connected":
    return 2
   if feat =="Separate apartment":
      return 3
   else:
      return 4
data["dweltyp"]=data["dweltyp"].apply(convert6)
def convert7(feat):
   if pd.isna(feat):
      return 0
   if feat =="Employed":
      return 1
   else:
      return 2
data["employed"]=data["employed"].apply(convert7)
def convert8(feat):
   if pd.isna(feat):
      return 0
   if  feat =="Complete Primary Education":
      return 1
   if feat =="Incomplete Primary Education":
      return 2
   if feat =="Complete Secondary Education":
      return 3
   if feat =="Incomplete Secondary Education":
      return 4
   if feat =="Complete Tertiary Education":
      return 5
   if feat =="Incomplete Tertiary Education":
      return  6
data['educ_max']=data["educ_max"].apply(convert8)
def convert9(feat):
   if pd.isna(feat):
      return 0
   if feat =="Yes":
      return 1
   else:
      return 2
data["any_nonagric"]=data["any_nonagric"].apply(convert9)
def convert10(feat):
   if pd.isna(feat):
      return 0
   if feat =="Transport, storage and communications":
      return 1
   if feat=="Public administration and defence":
      return 2
   if feat =="Construction":
      return 3
   if feat =="Manufacturing":
      return 4
   if feat =="Wholesale and retail trade":
      return 5
   if feat =="Education":
      return 6
   if feat=="Agriculture, hunting, forestry and fishing":
      return 7
   if feat=="Health and social work":
      return 8
   if feat=="Other services":
      return 9
   if feat=="Hotels and restaurants":
      return 10
   if feat=="Financial intermediation":
      return 11
   if feat=="Real estate, renting and business activities": 
      return 12
   if feat=="Community, social and personal services":
      return 13
   if feat=="Electricity, gas and water supply":
      return 14
   
   if feat=="Mining and quarrying":
      return 15
   else:
      return 16   
data["sector1d"]=data["sector1d"].apply(convert10)
def convert11(feat):
   if feat =="Urban":
      return 1
   else:
      return 2
data["urban"]=data["urban"].apply(convert11)
  
consumed_units = [f"consumed{i*100}" for i in range(1, 51)]

for col in consumed_units:
    if col in data.columns:
        try:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            data[col] = data[col].fillna(0)
            data[col] = data[col].astype(int)
        except Exception as e:
            print(f"Error converting column {col}: {e}")
print(data.head())

print("\nOutlier Detection and Removal")
print(f"Dataset shape before outlier removal: {data.shape}")

numerical_columns = data.select_dtypes(include=[np.number]).columns
print(f"Numerical columns for outlier detection: {len(numerical_columns)}")

plt.figure(figsize=(20, 15))
for i, col in enumerate(numerical_columns[:12]):
    plt.subplot(3, 4, i+1)
    plt.boxplot(data[col].dropna())
    plt.title(f'{col} - Before Outlier Removal')
    plt.xticks([])
plt.tight_layout()
plt.suptitle('Boxplots Before Outlier Removal', fontsize=16)
plt.savefig('boxplots_before_outlier_removal.png', dpi=150, bbox_inches='tight')
plt.close()  # Close to free memory

original_data = data.copy()

outliers_removed = 0
for col in numerical_columns:
    if col in data.columns:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_count = len(data[(data[col] < lower_bound) | (data[col] > upper_bound)])
        outliers_removed += outliers_count
        
        if outliers_count > 0:
            print(f"Column '{col}': {outliers_count} outliers detected (< {lower_bound:.2f} or > {upper_bound:.2f})")
        
        data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]

print(f"\nTotal outliers removed: {outliers_removed}")
print(f"Dataset shape after outlier removal: {data.shape}")
print(f"Data reduction: {((original_data.shape[0] - data.shape[0]) / original_data.shape[0]) * 100:.2f}%")

plt.figure(figsize=(20, 15))
for i, col in enumerate(numerical_columns[:12]):
    if col in data.columns:
        plt.subplot(3, 4, i+1)
        plt.boxplot(data[col].dropna())
        plt.title(f'{col} - After Outlier Removal')
        plt.xticks([])
plt.tight_layout()
plt.suptitle('Boxplots After Outlier Removal', fontsize=16)
plt.savefig('boxplots_after_outlier_removal.png', dpi=150, bbox_inches='tight')
plt.close()  # Close to free memory

print("\nModel Training")
from sklearn.model_selection import train_test_split
X = data.drop("utl_exp_ppp17", axis=1, errors="ignore")
y = data["utl_exp_ppp17"]
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# Split training data for XGBoost validation BEFORE scaling
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_split = scaler.fit_transform(X_train_split)
X_val_split = scaler.transform(X_val_split)
X_test = scaler.transform(X_test)

import xgboost as xgb

# XGBoost model with early stopping built into parameters
xgb_model = xgb.XGBRegressor(
   n_estimators=1000,  
   eval_metric='rmse',
   early_stopping_rounds=50  
)
# Option 1: Simple early stopping without GridSearch
print("Training XGBoost with Early Stopping...")
xgb_model.fit(
   X_train_split, y_train_split,
   eval_set=[(X_val_split, y_val_split)],
   verbose=True
)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Make predictions
y_pred_xgb = xgb_model.predict(X_test)

# Calculate metrics
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)

print(f"XGBoost Results:")
print(f"Mean Squared Error: {mse_xgb:.4f}")
print(f"Mean Absolute Error: {mae_xgb:.4f}")
print(f"R-squared: {r2_xgb:.4f}")
print(f"Best iteration: {xgb_model.best_iteration if hasattr(xgb_model, 'best_iteration') else 'N/A'}")

# Model saving section
import joblib
joblib.dump(scaler, "scaler.pkl")
joblib.dump(xgb_model, "xgb_poverty_model.pkl")  

# Also create shapes info for compatibility
shapes = {
    "num_features": X_train_split.shape[1]
}
joblib.dump(shapes, "input_shapes.pkl")

# Save feature metadata for the web application
feature_names = list(X.columns)
feature_defaults = {}
for col in feature_names:
    feature_defaults[col] = float(X[col].median())
feature_meta = {
    "feature_names": feature_names,
    "feature_defaults": feature_defaults,
}
joblib.dump(feature_meta, "feature_meta.pkl")
print("XGBoost model, scaler, and feature metadata saved successfully.")

print("\nPrediction Analysis")
print(f"Number of test samples: {len(y_test)}")
print(f"Actual values range: {y_test.min():.2f} to {y_test.max():.2f}")
print(f"Predicted values range: {y_pred_xgb.min():.2f} to {y_pred_xgb.max():.2f}")
print(f"Mean actual value: {y_test.mean():.2f}")
print(f"Mean predicted value: {y_pred_xgb.mean():.2f}")

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(y_test, y_pred_xgb, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Predicted vs Actual Values')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
residuals = y_test.values.flatten() - y_pred_xgb.flatten()
plt.scatter(y_pred_xgb, residuals, alpha=0.6, color='green')
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.hist(y_pred_xgb, bins=30, alpha=0.7, label='Predicted', color='blue')
plt.hist(y_test, bins=30, alpha=0.7, label='Actual', color='red')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Distribution of Predictions vs Actual')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('prediction_analysis.png', dpi=150, bbox_inches='tight')
plt.close()  # Close to free memory
print("Prediction analysis plots saved as 'prediction_analysis.png'")

print(f"\nSample Predictions")
sample_indices = np.random.choice(len(y_test), size=10, replace=False)
print(f"{'Index':<8} {'Actual':<12} {'Predicted':<12} {'Error':<12} {'Error %':<10}")
print("-" * 55)
for idx in sample_indices:
    actual = y_test.iloc[idx]
    predicted = y_pred_xgb[idx]
    error = abs(actual - predicted)
    error_pct = (error / actual) * 100 if actual != 0 else 0
    print(f"{idx:<8} {actual:<12.2f} {predicted:<12.2f} {error:<12.2f} {error_pct:<10.1f}%")

def predict_poverty_level(new_data_dict):
    """Predict poverty level for new household data"""
    try:
        new_data = pd.DataFrame([new_data_dict])
        new_data_scaled = scaler.transform(new_data)
        prediction = xgb_model.predict(new_data_scaled)
        return prediction[0]  
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

def create_sample_household():
    """Create a sample household for prediction demonstration"""
    sample_idx = np.random.choice(len(X_train_split))
    sample_features = scaler.inverse_transform(X_train_split[sample_idx:sample_idx+1])
    feature_names = X.columns
    sample_dict = {}
    for i, feature in enumerate(feature_names):
        sample_dict[feature] = sample_features[0][i]
    return sample_dict

print(f"\nExample Prediction")
sample_household = create_sample_household()
print("Sample household features (first 10):")
for i, (key, value) in enumerate(list(sample_household.items())[:5]):
    print(f"  {key}: {value:.2f}")
print("  ... (showing first 5 features only)")

sample_prediction = predict_poverty_level(sample_household)
if sample_prediction is not None:
    print(f"\nPredicted poverty level (utility expenditure): {sample_prediction:.2f}")

def load_trained_model():
    """Load the saved model and scaler for predictions"""
    try:
        import joblib
        
        # Try to load XGBoost model first
        try:
            xgb_model = joblib.load("xgb_poverty_model.pkl")
            scaler = joblib.load("scaler.pkl")
            print("XGBoost model loaded successfully!")
            return xgb_model, scaler, None
        except Exception:
            print("Could not load model files.")
            return None, None, None
            
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None

print(f"\nModel Summary:")
print(f"✓ Model trained and saved")
print(f"✓ Scaler saved for preprocessing")
print(f"✓ Prediction function created")
print(f"✓ Model evaluation completed")

print(data.info())
