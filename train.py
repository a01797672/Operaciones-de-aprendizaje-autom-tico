import argparse
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

import mlflow
import mlflow.sklearn

def set_seed(seed):
    """Fija la semilla aleatoria para garantizar reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)

def load_and_preprocess_data(data_path):
    """Carga el dataset, limpia duplicados y columnas."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"No se encontró el archivo de datos en: {data_path}")
    
    # Carga con separador ';' y limpia comillas si las hay
    df = pd.read_csv(data_path, sep=';', engine="python")
    df.columns = df.columns.str.replace('"', '')
    
    # Eliminar duplicados
    df = df.drop_duplicates()
    
    X = df.drop(columns=['quality'])
    y = df['quality']
    return X, y

def generate_artifacts(y_test, predictions):
    """Genera artefactos visuales para el registro en MLFlow."""
    os.makedirs("artifacts", exist_ok=True)
    artifact_path = "artifacts/actual_vs_predicted.png"
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=predictions, alpha=0.5)
    plt.xlabel("Calidad Real")
    plt.ylabel("Calidad Predicha")
    plt.title("Valores Reales vs Predichos (Regresión Lineal)")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
    plt.savefig(artifact_path)
    plt.close()
    
    return artifact_path

def main():
    parser = argparse.ArgumentParser(description="Script de entrenamiento parametrizado con MLFlow para calidad de vino.")
    parser.add_argument("--data_path", type=str, default="winequality-red.csv", help="Ruta hacia el archivo CSV del dataset.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Proporción del conjunto de prueba.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla aleatoria para la reproducibilidad.")
    parser.add_argument("--fit_intercept", type=lambda x: (x.lower() == 'true'), default=True, help="Si se debe calcular el intercepto (True/False).")
    
    args = parser.parse_args()
    
    # 1. Fijar semilla
    set_seed(args.seed)
    
    # 2. Configurar explícitamente el tracking uri local y el experimento limpio
    mlflow.set_tracking_uri("./mlruns")
    mlflow.set_experiment("wine-quality-baseline")
    
    with mlflow.start_run():
        # Registrar hiperparámetros
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("seed", args.seed)
        mlflow.log_param("fit_intercept", args.fit_intercept)
        
        # Carga y preprocesamiento
        X, y = load_and_preprocess_data(args.data_path)
        
        # División train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.seed
        )
        
        # Entrenamiento
        model = LinearRegression(fit_intercept=args.fit_intercept)
        model.fit(X_train, y_train)
        
        # Evaluación
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)
        
        # Registrar métricas
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)
        
        # Generar y registrar artefactos (gráfico)
        artifact_path = generate_artifacts(y_test, predictions)
        mlflow.log_artifact(artifact_path)
        
        # Registrar el modelo entrenado con el parámetro moderno de MLFlow
        mlflow.sklearn.log_model(
            model, 
            name="model",
            pip_requirements=["scikit-learn", "pandas", "numpy", "matplotlib", "seaborn"]
        )
        
        print(f"Corrida finalizada (Semilla {args.seed}). RMSE: {rmse:.4f} | R2: {r2:.4f}")

if __name__ == "__main__":
    main()