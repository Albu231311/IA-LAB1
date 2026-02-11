import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score

def run_comparison():
    # --- 1. Generación de Datos ---
    # Usamos make_moons para generar datos no lineales y no convexos
    # random_state fijo para reproducibilidad
    X, y_true = make_moons(n_samples=200, noise=0.08, random_state=42)

    # --- 2. Definición de Modelos ---
    models = {
        'K-Means': KMeans(n_clusters=2, random_state=42, n_init=10),
        'Agglomerative (Ward)': AgglomerativeClustering(n_clusters=2, linkage='ward'),
        'Agglomerative (Complete)': AgglomerativeClustering(n_clusters=2, linkage='complete'),
        'Agglomerative (Average)': AgglomerativeClustering(n_clusters=2, linkage='average'),
        'Agglomerative (Single)': AgglomerativeClustering(n_clusters=2, linkage='single')
    }

    results = []
    
    # Configuración de visualización
    n_models = len(models)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    # Plot Ground Truth
    axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', s=40, edgecolor='k')
    axes[0].set_title("Ground Truth (Estructura Real)")
    axes[0].set_xlabel("X1")
    axes[0].set_ylabel("X2")

    # --- 3. Ejecución y Evaluación ---
    for idx, (name, model) in enumerate(models.items(), start=1):
        ax = axes[idx]
        
        # Ajuste y Predicción
        try:
            y_pred = model.fit_predict(X)
        except Exception as e:
            print(f"Error ejecutando {name}: {e}")
            continue

        # Métricas
        # ARI: Mide similitud con la verdad (1.0 es perfecto, 0.0 es aleatorio)
        ari = adjusted_rand_score(y_true, y_pred)
        # Silhouette: Mide cohesión y separación (no siempre ideal para formas no convexas)
        sil = silhouette_score(X, y_pred)
        
        results.append({
            'Algoritmo': name,
            'ARI (Fidelidad)': ari,
            'Silhouette': sil,
            'Linkage/Método': name.split('(')[1].replace(')', '') if '(' in name else 'Centroid-based'
        })
        
        # Visualización
        ax.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', s=40, edgecolor='k')
        
        # Comentarios sobre K-Means y convexidad
        # K-means calculca centroides y asigna puntos al más cercano (Euclidiana).
        # Esto crea fronteras de decisión lineales (Voronoi), forzando clusters convexos.
        # Por eso "corta" las lunas por la mitad en lugar de seguirlas.
        if "K-Means" in name:
            centers = model.cluster_centers_
            ax.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Centroides')
            ax.legend(loc='upper right')
            
        ax.set_title(f"{name}\nARI: {ari:.3f} | Sil: {sil:.3f}")
        ax.set_xlabel("X1")
        ax.set_ylabel("X2")

    plt.tight_layout()
    plt.show()

    # --- 4. Organización de Resultados ---
    df_results = pd.DataFrame(results).set_index('Algoritmo')
    
    print("\n" + "="*60)
    print("COMPARACIÓN DE K-MEANS VS AGGLOMERATIVE CLUSTERING (Datos No Lineales)")
    print("="*60)
    print(df_results)
    print("="*60)


run_comparison()
