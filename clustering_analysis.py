import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import numpy as np
import os

file_path = 'countries_binary.xlsx'
k_clusters = 3

def run_analysis():
    print("\n=== 3. Hacer un agrupamiento jerarquico con los datos de los paises countries binary.xlsx ===")
    print(f"Procesando archivo: {file_path}")

    try:
        df = pd.read_excel(file_path)

        labels = df.iloc[:, 0].values
        data = df.iloc[:, 1:]
            
        print(f"Datos cargados.")
    except Exception as e:
        print(f"Error al leer Excel: {e}")
        return

    methods = ['single', 'complete', 'average', 'ward']
    metrics = ['euclidean', 'Hamming']

    output_dir = "dendrograms"
    os.makedirs(output_dir, exist_ok=True)

    print("\n--- Generando Dendrogramas ---")
    for method in methods:
        for metric in metrics:
            if method == 'ward' and metric != 'euclidean':
                continue

            try:
                plt.figure(figsize=(12, 6))
                plt.title(f"Dendrograma (Método: {method}, Métrica: {metric})")
                
                Z = linkage(data, method=method, metric=metric)
                
                dendrogram(Z, labels=labels, leaf_rotation=90)
                plt.tight_layout()
                
                plot_filename = os.path.join(output_dir, f"dendrogram_{method}_{metric}.png")
                plt.savefig(plot_filename)
                plt.close()
                print(f"Guardado: {plot_filename}")
                
            except Exception as e:
                print(f"Error en jerárquico ({method}, {metric}): {e}")

    print("\n=== 4. Realizar un analisis de agrupamiento k-means, nuevamente para los datos de los paises, que estan disponibles en el archivo \ncountries binary.xlsx. Contrastar con el ejercicio anterior. ¿Son iguales las agrupaciones? ¿Por qué? Justificar. ===")
    print(f"\n--- Análisis K-Means (K={k_clusters}) ---")
    kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
    kmeans.fit(data)
    kmeans_labels = kmeans.labels_
    
    print("Etiquetas asignadas por K-Means:")
    print(kmeans_labels)

    print("\n--- Contraste: K-Means vs Jerárquico (Ward) ---")
    
    Z_ward = linkage(data, method='ward', metric='euclidean')
    hierarchical_labels = fcluster(Z_ward, k_clusters, criterion='maxclust')
    
    score = adjusted_rand_score(kmeans_labels, hierarchical_labels)
    
    print(f"Índice Adjusted Rand (Similitud): {score:.4f}")

if __name__ == "__main__":
    run_analysis()
