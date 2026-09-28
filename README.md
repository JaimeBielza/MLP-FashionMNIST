# Lab 2: red fully connected sobre FashionMNIST (PyTorch)

Este laboratorio estudia cómo se comporta una red neuronal totalmente conectada (MLP) al clasificar las imágenes de ropa de **FashionMNIST**. Se entrena con PyTorch y después se analiza en qué acierta, en qué falla y por qué.

## El dataset

[FashionMNIST](https://github.com/zalandoresearch/fashion-mnist) (Zalando) es un sustituto de MNIST más difícil que el original:

- 60 000 imágenes de entrenamiento y 10 000 de test.
- Imágenes de 28×28 píxeles en escala de grises.
- 10 clases equilibradas:

| Índice | Clase       | Índice | Clase      |
|:------:|-------------|:------:|------------|
| 0      | T-shirt/top | 5      | Sandal     |
| 1      | Trouser     | 6      | Shirt      |
| 2      | Pullover    | 7      | Sneaker    |
| 3      | Dress       | 8      | Bag        |
| 4      | Coat        | 9      | Ankle boot |

La primera vez que se ejecuta el script, `torchvision` descarga el dataset en `data/FashionMNIST/`.

## Arquitectura

La imagen se aplana a un vector de 784 valores y pasa por dos capas ocultas de 512 neuronas con ReLU y Dropout:

```
Input (1×28×28)
  └─ Flatten                    → 784
  └─ Linear(784, 512) + ReLU + Dropout(0.3)
  └─ Linear(512, 512) + ReLU + Dropout(0.5)
  └─ Linear(512, 10)            → logits (una por clase)
```

- **Parámetros entrenables:** 669 706.
- El Dropout es más agresivo en la segunda capa (0.5) para frenar el sobreajuste, que en un MLP de este tamaño aparece enseguida.
- La salida son logits sin softmax, porque `CrossEntropyLoss` ya aplica internamente `log_softmax`.

## Configuración del entrenamiento

| Parámetro      | Valor                                   |
|----------------|-----------------------------------------|
| Función de pérdida | `nn.CrossEntropyLoss`               |
| Optimizador    | Adam, `lr = 1e-3`                       |
| Batch size     | 64 (`shuffle=True`, `drop_last=True` en train) |
| Épocas         | 50                                      |
| Preprocesado   | `ToTensor()` (normaliza píxeles a [0, 1]) |
| Dispositivo    | `mps` (GPU Apple Silicon) si está disponible; si no, `cpu` |

En cada época se entrena sobre el conjunto completo y después se evalúa sobre test, mostrando la accuracy y la pérdida media. Al terminar se guardan los pesos en `fashion_mnist_model.pth` y se hace una predicción de ejemplo sobre la primera imagen de test.

## Resultados

Evaluación del modelo guardado (`fashion_mnist_model.pth`) sobre las 10 000 imágenes de test:

**Accuracy global: 89.6 %**

### Accuracy por clase

| Clase       | Accuracy |
|-------------|:--------:|
| Trouser     | 98.1 %   |
| Sandal      | 97.7 %   |
| Sneaker     | 97.5 %   |
| Bag         | 97.2 %   |
| Ankle boot  | 95.5 %   |
| Dress       | 93.0 %   |
| T-shirt/top | 86.9 %   |
| Pullover    | 83.8 %   |
| Coat        | 80.5 %   |
| **Shirt**   | **66.2 %** |

### Confusiones más frecuentes

| Real      | Predicho    | Nº de errores |
|-----------|-------------|:-------------:|
| Shirt     | T-shirt/top | 139 |
| Coat      | Pullover    | 113 |
| Shirt     | Pullover    | 95  |
| Pullover  | Coat        | 78  |
| T-shirt/top | Shirt     | 78  |

## Análisis del comportamiento

**Lo que la red resuelve bien.** Pantalones, bolsos y calzado (sandalias, zapatillas y botines) superan el 95 %. Su silueta es muy distinta de la del resto, así que la distribución de píxeles basta para separarlos sin necesidad de entender formas locales.

**Dónde falla.** Casi todos los errores se concentran en las prendas de parte superior: camisa, camiseta, jersey y abrigo. A 28×28 píxeles las cuatro tienen un contorno casi idéntico (tronco más mangas) y lo que las distingue son detalles como el cuello, los botones, la cremallera o la textura. La clase *Shirt* es la peor (66 %) porque visualmente queda en medio: se confunde con *T-shirt/top* por un lado y con *Pullover* por otro.

**La limitación de fondo es la arquitectura.** Al hacer `Flatten`, la red trata la imagen como una lista de 784 números independientes y pierde la estructura espacial: no sabe qué píxeles son vecinos. Tampoco tiene invarianza a pequeños desplazamientos. Por eso aprende bien las formas globales y mal los detalles locales. Un MLP así se queda normalmente en torno al 88-90 % en FashionMNIST, mientras que una CNN sencilla pasa del 91-93 % justo porque captura esos patrones locales.

**Regularización.** Con cerca de 670 000 parámetros y 50 épocas, la red puede memorizar el conjunto de entrenamiento. El Dropout (0.3 y 0.5) mitiga ese sobreajuste, pero conviene fijarse en la evolución de la pérdida de test por época: si se estanca o sube mientras la de entrenamiento sigue bajando, la red está sobreajustando y sobran épocas.


## Estructura del proyecto

```
lab2/
├── nnMNIST.py                # Modelo, carga de datos, entrenamiento y test
├── fashion_mnist_model.pth   # Pesos del modelo entrenado (state_dict)
├── data/FashionMNIST/        # Dataset (se descarga automáticamente)
└── venv/                     # Entorno virtual de Python
```

## Cómo ejecutarlo

Requisitos: Python 3.14, `torch` 2.14 y `torchvision` 0.29 (las versiones usadas en el `venv`).

```bash
python -m venv venv
source venv/bin/activate
pip install torch torchvision

python nnMNIST.py
```

El script entrena, va mostrando la pérdida y la accuracy de cada época, guarda el modelo e imprime una predicción de ejemplo.

### Cargar el modelo ya entrenado

```python
import torch
from nnMNIST import NeuralNetwork

model = NeuralNetwork(28 * 28)
model.load_state_dict(torch.load("fashion_mnist_model.pth", map_location="cpu"))
model.eval()
```

> **Nota:** las funciones `train()` y `test()` usan la variable global `device`, que solo se define dentro de `if __name__ == '__main__'`. Si se importan desde otro script, hay que definir `device` antes o pasarlo como argumento.
