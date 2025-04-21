import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os
from datetime import datetime

def train_model():
    # Load dataset
    data = pd.read_csv('C:/Users/rahul/OneDrive/Desktop/genai/Sales Prediction/backend/data/advertising.csv')  # Adjust path if needed
    X = data[['TV', 'Radio', 'Newspaper']]
    y = data['Sales']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model with Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict on test set
    y_pred = model.predict(X_test)

    # Evaluate model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    print(f"R^2 Score: {r2}")

    # Save model to disk with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(os.path.dirname(__file__), f'model_{timestamp}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model trained and saved as {save_path}!")

if __name__ == "__main__":
    train_model()