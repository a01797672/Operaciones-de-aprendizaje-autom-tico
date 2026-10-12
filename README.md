# Proyecto: Predicción de Calidad de Vino Rojo (MLOps)

Este repositorio contiene la solución modularizada para el entrenamiento, seguimiento de experimentos y evaluación de un modelo de regresión para predecir la calidad de vino rojo.

## Archivos del Repositorio
- `train.py`: Script principal de entrenamiento modularizado e instrumentado con **MLFlow**.
- `requirements.txt`: Archivo de dependencias con las versiones exactas del entorno de ejecución.
- `winequality-red.csv`: Dataset utilizado para el modelo.

## Requisitos Previos e Instalación
Para instalar las dependencias necesarias en un entorno local, ejecuta:
pip install -r requirements.txt

Instrucciones de Ejecución
El script train.py está parametrizado mediante argparse. Puedes ejecutarlo directamente con los valores por defecto:
python train.py

Se puede modificar los hiperparámetros desde la línea de comandos (por ejemplo, cambiando la semilla aleatoria, el tamaño del conjunto de prueba o la interceptación):
python train.py --seed 42 --test_size 0.2
python train.py --seed 123 --test_size 0.25
python train.py --seed 777 --fit_intercept False

Visualización de Experimentos con MLFlow
Para abrir la interfaz gráfica de MLFlow y comparar las métricas y artefactos de las diferentes corridas:
Ejecuta el servidor local de MLFlow:
mlflow ui
Abre tu navegador e ingresa a la siguiente dirección:
http://127.0.0.1:5000
