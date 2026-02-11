import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import data
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def quantize_image(image, n_colors=8):
    """
    Realiza la cuantización de colores de una imagen utilizando K-Means.
    
    Args:
        image: Imagen RGB (numpy array HxWx3).
        n_colors: Número de colores (clusters) a reducir.
        
    Returns:
        label_map: Mapa de clases (HxW) donde cada pixel es el índice del cluster.
        quantized_img: Imagen re-coloreada con los centroides (HxWx3).
    """
    # 1. Preprocesamiento
    # Normalizamos a 0-1 si viene en 0-255 (estándar para matplotlib floats)
    if image.dtype == np.uint8:
        img_data = image / 255.0
    else:
        img_data = image.copy()
        
    h, w, d = img_data.shape
    
    # Transformar a una lista de pixeles (N_muestras, N_features=3)
    # Aplanamos la imagen para que cada pixel sea una observación
    pixels = img_data.reshape((h * w, d))
    
    # 2. Clustering (K-Means)
    # n_init=10 reduce la aleatoriedad ejecutando el algoritmo 10 veces y eligiendo el mejor
    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
    
    # Entrenamos y obtenemos las etiquetas para cada pixel
    # fit_predict es equivalente a fit() seguido de labels_
    labels_flat = kmeans.fit_predict(pixels)
    centers = kmeans.cluster_centers_
    
    # 3. Reconstrucción
    # Mapa de Clases: Reconstruimos la forma 2D de la imagen con las etiquetas
    label_map = labels_flat.reshape((h, w))
    
    # Imagen Cuantizada: Asignamos a cada pixel el color de su centroide
    quantized_flat = centers[labels_flat]
    quantized_img = quantized_flat.reshape((h, w, d))
    
    return label_map, quantized_img

def main():
    # --- 1. Selección de Imágenes ---
    # Usamos imágenes estándar de Scikit-Image para asegurar reproducibilidad
    # Si desea usar sus propias imágenes, puede cargarlas con plt.imread('ruta/imagen.jpg')
    print("Cargando imágenes...")
    
    images_to_process = [
        ("Astronauta", data.astronaut()),      # Imagen rica en color y rostro
        ("Gato (Chelsea)", data.chelsea()),    # Texturas y pelaje
        ("Café (Taza)", data.coffee())         # Objetos y contraste
    ]
    
    K_COLORS = 8  # Número de colores para la cuantización
    
    # Configuración del plot 3x3
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    plt.suptitle(f"Cuantización de Colores usando K-Means (K={K_COLORS})", fontsize=16)
    
    print(f"Procesando {len(images_to_process)} imágenes con K={K_COLORS} colores...")

    for idx, (name, img) in enumerate(images_to_process):
        print(f"  - Procesando: {name}...")
        
        # Ejecutar algoritmo
        label_map, quantized_img = quantize_image(img, n_colors=K_COLORS)
        
        # --- Visualización ---
        # Columna 1: Imagen Original
        ax_orig = axes[idx, 0]
        ax_orig.imshow(img)
        ax_orig.set_title(f"{name} - Original")
        ax_orig.axis('off')
        
        # Columna 2: Mapa de Clases (Clusters)
        ax_map = axes[idx, 1]
        # Usamos un mapa de colores discreto (tab10 o tab20) para distinguir las clases
        im_map = ax_map.imshow(label_map, cmap='tab20')
        ax_map.set_title(f"Mapa de Clases / Agrupamiento")
        ax_map.axis('off')
        # Barra de color opcional para entender los índices
        # plt.colorbar(im_map, ax=ax_map, fraction=0.046, pad=0.04)
        
        # Columna 3: Imagen Cuantizada
        ax_quant = axes[idx, 2]
        ax_quant.imshow(quantized_img)
        ax_quant.set_title(f"Imagen Cuantizada ({K_COLORS} colores)")
        ax_quant.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Ajuste para el título general
    print("Mostrando resultados...")
    plt.show()

if __name__ == "__main__":
    main()
