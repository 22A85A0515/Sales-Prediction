import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def load_and_preprocess_data():
    # Load dataset
    data = pd.read_csv('C:/Users/rahul/OneDrive/Desktop/genai/Sales Prediction/backend/data/advertising.csv')
    
    # Fill missing values with the mean (if any)
    data = data.fillna(data.mean())
    
    return data

def train_model(X, y):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a Random Forest Regressor model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return mse, r2, y_pred

def plot_results(data, y_test, y_pred, model, feature_names):
    # Create directory for saving plots
    os.makedirs('frontend/images', exist_ok=True)
    
    # 1. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title('Feature Correlation Heatmap')
    plt.savefig('frontend/images/correlation.png')
    plt.show()  # Display the plot
    plt.close()
    
    # 2. Distribution of Sales
    plt.figure(figsize=(10, 6))
    sns.histplot(data['Sales'], kde=True, color='blue')
    plt.title('Distribution of Sales')
    plt.xlabel('Sales')
    plt.ylabel('Frequency')
    plt.savefig('frontend/images/sales_distribution.png')
    plt.show()  # Display the plot
    plt.close()
    
    # 3. Actual vs Predicted Sales
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='green')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Diagonal line
    plt.xlabel('Actual Sales')
    plt.ylabel('Predicted Sales')
    plt.title('Actual vs Predicted Sales')
    plt.savefig('frontend/images/actual_vs_predicted.png')
    plt.show()  # Display the plot
    plt.close()
    
    # 4. Residual Plot
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals, color='red')
    plt.axhline(y=0, color='black', linestyle='--')
    plt.xlabel('Predicted Sales')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.savefig('frontend/images/residual_plot.png')
    plt.show()  # Display the plot
    plt.close()
    
    # 5. Feature Importance (Random Forest)
    plt.figure(figsize=(10, 6))
    feature_importances = pd.Series(model.feature_importances_, index=feature_names)
    feature_importances.sort_values(ascending=False).plot(kind='bar', color='purple')
    plt.title('Feature Importance (Random Forest)')
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.savefig('frontend/images/feature_importance.png')
    plt.show()  # Display the plot
    plt.close()

def main():
    # Load and preprocess data
    data = load_and_preprocess_data()
    
    # Define features (X) and target (y)
    X = data[['TV', 'Radio', 'Newspaper']]
    y = data['Sales']
    
    # Train model
    model, X_test, y_test = train_model(X, y)
    
    # Evaluate model
    mse, r2, y_pred = evaluate_model(model, X_test, y_test)
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared Score: {r2}")
    
    # Plot results
    plot_results(data, y_test, y_pred, model, feature_names=X.columns)
    
    # Save model
    joblib.dump(model, 'C:/Users/rahul/OneDrive/Desktop/genai/Sales Prediction/backend/model_random_forest.pkl')

if __name__ == "__main__":
    main()