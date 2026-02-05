import random

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
                nuevo_centroide = [sum(p[dim] for p in puntos_cluster) / len(puntos_cluster) for dim in range(d)]
                nuevos_centroides.append(nuevo_centroide)
            else:
                nuevos_centroides.append(centroides[j])
        
        if not cambio_en_labels:
            break
        centroides = nuevos_centroides
        
    return labels, centroides


#Entrada de datos

print("K-Means desde cero")

n = int(input("Ingrese el número de datos (n): "))
d = int(input("Ingrese la dimensión de los vectores (d): "))

matriz_datos = []
for i in range(n):
    fila_raw = input(f"Ingrese los {d} valores del dato {i+1} separados por espacio: ")
    # Convertimos el string de entrada en una lista de flotantes
    fila = [float(x) for x in fila_raw.split()]
    matriz_datos.append(fila)

k = int(input("Ingrese el número de clusters (k): "))

# Ejecución
clases, centroides_finales = k_means_custom(matriz_datos, k)

# --- SALIDA ---
print("\n RESULTADOS")
print(f"Vector de etiquetas (clases): {clases}")
print("Matriz de centroides (k x d):")
for i, c in enumerate(centroides_finales):
    print(f" Centroide {i}: {c}")