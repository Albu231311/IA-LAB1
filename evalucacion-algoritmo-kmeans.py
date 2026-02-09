import random
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def calcular_distancia_euclidiana(punto_a, punto_b):
    suma_cuadrados = sum((a - b) ** 2 for a, b in zip(punto_a, punto_b))
    return suma_cuadrados ** 0.5

def k_means_custom(datos, k, max_iter=100):
    n = len(datos)
    d = len(datos[0])
    
    # Inicialización aleatoria
    centroides = random.sample(datos, k)
    labels = [0] * n
    
    for _ in range(max_iter):
        cambio_en_labels = False
        
        # PASO ASIGNACIÓN
        for i in range(n):
            distancias = [calcular_distancia_euclidiana(datos[i], c) for c in centroides]
            nueva_label = distancias.index(min(distancias))
            if labels[i] != nueva_label:
                labels[i] = nueva_label
                cambio_en_labels = True
        
        # PASO ACTUALIZACIÓN
        nuevos_centroides = []
        for j in range(k):
            puntos_cluster = [datos[i] for i in range(n) if labels[i] == j]
            if puntos_cluster:
                nuevo_centroide = [
                    sum(p[dim] for p in puntos_cluster) / len(puntos_cluster)
                    for dim in range(d)
                ]
                nuevos_centroides.append(nuevo_centroide)
            else:
                nuevos_centroides.append(centroides[j])
        
        if not cambio_en_labels:
            break
        
        centroides = nuevos_centroides
        
    return labels, centroides

# DATASET 1: IRIS

print("\n IRIS ")

iris = load_iris()
X_iris = iris.data

# Normalización
scaler = StandardScaler()
X_iris = scaler.fit_transform(X_iris)

k = 3 #numero de clusters

# K-means desde cero
labels_iris_custom, centroides_iris_custom = k_means_custom(X_iris.tolist(), k)

# K-means scikit-learn
kmeans_iris_sklearn = KMeans(n_clusters=k, random_state=42, n_init=10)
labels_iris_sklearn = kmeans_iris_sklearn.fit_predict(X_iris)

print("K-means desde cero (primeros 10 labels):", labels_iris_custom[:10])
print("K-means scikit-learn (primeros 10 labels):", labels_iris_sklearn[:10])

# DATASET 2: PENGUINS

print("\n PENGUINS ")

penguins = sns.load_dataset("penguins")
penguins = penguins.dropna()

X_penguins = penguins[
    ['bill_length_mm', 'bill_depth_mm',
     'flipper_length_mm', 'body_mass_g']
].values

# Normalización
X_penguins = scaler.fit_transform(X_penguins)

k = 3 #numero de clusters

# K-means desde cero
labels_penguins_custom, _ = k_means_custom(X_penguins.tolist(), k)

# K-means scikit-learn
kmeans_penguins_sklearn = KMeans(n_clusters=k, random_state=42, n_init=10)
labels_penguins_sklearn = kmeans_penguins_sklearn.fit_predict(X_penguins)

print("K-means desde cero (primeros 10 labels):", labels_penguins_custom[:10])
print("K-means scikit-learn (primeros 10 labels):", labels_penguins_sklearn[:10])

# DATASET 3: WINE QUALITY (RED)

print("\n WINE ")

df_wine = pd.read_csv("winequality-red.csv", sep=";")
X_wine = df_wine.values

# Normalización
X_wine = scaler.fit_transform(X_wine)

k = 3 #numero de clusters

# K-means desde cero
labels_wine_custom, _ = k_means_custom(X_wine.tolist(), k)

# K-means scikit-learn
kmeans_wine_sklearn = KMeans(n_clusters=k, random_state=42, n_init=10)
labels_wine_sklearn = kmeans_wine_sklearn.fit_predict(X_wine)

print("K-means desde cero (primeros 10 labels):", labels_wine_custom[:10])
print("K-means scikit-learn (primeros 10 labels):", labels_wine_sklearn[:10])

print("\nEjecución finalizada correctamente.")
