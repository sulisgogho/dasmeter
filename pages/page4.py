import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import pickle

from navigation import make_sidebar

make_sidebar()

# Function to load data and separate features and target column
def load_data(df, target_column):
    # Drop non-feature columns and select the specified target
    X = df.drop(['Nama', 'Umur', 'Jenis Kelamin', 'DEPRESSION', 'ANXIETY', 'STRESS'], axis=1)
    y = df[target_column]
    return X, y

# Train and save model function with SMOTE for handling imbalance
def train_and_save_model(X, y, filename):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Handle missing values with imputation
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    
    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    # Train the KNN model
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_smote, y_train_smote)
    
    # Save the model
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    
    # Model evaluation
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    st.write(f"Akurasi Model {filename}: {accuracy*100:.2f}%")
    
    # Display confusion matrix
    cm = confusion_matrix(y_test, predictions, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    
    fig_cm, ax_cm = plt.subplots()
    disp.plot(ax=ax_cm)
    plt.title(f'Confusion Matrix for {filename}')
    st.pyplot(fig_cm)

    # PCA Visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_test)
    
    # Encode labels for visualization
    le = LabelEncoder()
    y_test_encoded = le.fit_transform(y_test)
    predictions_encoded = le.transform(predictions)
    
    fig_pca, ax_pca = plt.subplots(figsize=(8,6))
    scatter = ax_pca.scatter(X_pca[:, 0], X_pca[:, 1], c=predictions_encoded, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, ticks=range(len(le.classes_)), ax=ax_pca)
    plt.title(f'PCA of Test Data with KNN Predictions for {filename}')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    st.pyplot(fig_pca)

    return model, imputer  # Return the trained model and imputer

# Combined prediction function to apply all models on uploaded test data
def combined_predict(models, imputers, X_new):
    results = {}
    X_new_imputed = imputers[0].transform(X_new)  # Apply imputation on new data

    # Loop through each model for each target
    for target, model in models.items():
        predictions = model.predict(X_new_imputed)
        results[f"Klasifikasi Status {target}"] = predictions
    
    return pd.DataFrame(results)

# Main function
def main():
    st.title("Model Klasifikasi Depresi, Kecemasan, dan Stres Atlet")
    
    # Upload dataset for training
    uploaded_file = st.file_uploader("Upload file CSV", type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        models = {}
        imputers = {}

        # Train and save models for each target column
        for target_column in ['DEPRESSION', 'ANXIETY', 'STRESS']:
            st.write(f"Training model for {target_column}")
            X, y = load_data(df, target_column)
            model, imputer = train_and_save_model(X, y, f"model_klasifikasi_{target_column.lower()}.pkl")
            models[target_column] = model
            imputers[target_column] = imputer

        # Predict with uploaded test data for classifications
        if st.button("Klasifikasi Data Baru"):
            test_file = st.file_uploader("Upload dataset baru untuk prediksi:", type=['csv'])
            
            if test_file is not None:
                df_new = pd.read_csv(test_file)
                X_new = df_new.drop(['Nama', 'Umur', 'Jenis Kelamin', 'DEPRESSION', 'ANXIETY', 'STRESS'], axis=1)  # Exclude target and non-feature columns
                predictions_df = combined_predict(models, list(imputers.values()), X_new)
                
                # Combine with original test data for display
                result_df = pd.concat([df_new[['Nama', 'Umur', 'Jenis Kelamin']], predictions_df], axis=1)
                st.write("Hasil Klasifikasi:")
                st.write(result_df)

if __name__ == "__main__":
    main()
