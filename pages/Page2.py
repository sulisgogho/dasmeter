# training.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import pickle
from navigation import make_sidebar

make_sidebar()

# Fungsi untuk memuat dan memproses dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    X = df.drop(['Diagnosa', 'Nama', 'Umur', 'Jenis Kelamin'], axis=1)
    y = df['Diagnosa']
    return X, y

# Fungsi untuk melatih model KNN dan menyimpannya ke dalam file
def train_and_save_model(X, y, filename):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    
    # Buat dan latih model
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_imputed, y_train)
    
    # Hitung akurasi
    X_test_imputed = imputer.transform(X_test)
    predictions = model.predict(X_test_imputed)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Akurasi Model: {accuracy*100:.2f}%")
    
    # Simpan model ke file menggunakan pickle
    with open(filename, 'wb') as file:
        pickle.dump(model, file)

def main():
    # Load dataset
    X, y = load_data("stres.csv")

    # Latih dan simpan model
    train_and_save_model(X, y, "model_klasifikasi.pkl")

if __name__ == "__main__":
    main()
