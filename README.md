# Procesamiento de Imágenes con Python

Este repositorio contiene los proyectos desarrollados durante un curso de procesamiento de imágenes, utilizando diversas herramientas y bibliotecas como Python, OpenCV, TensorFlow, MNIST, entre otras. A través de estos proyectos, se logra recortar, aplicar filtros, dibujar y editar imágenes, incluso usando la vista previa de la cámara.

## Contenido

- [Descripción](#descripción)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso](#uso)
- [Proyectos](#proyectos)
- [Contribuciones](#contribuciones)
- [Licencia](#licencia)

## Descripción

El curso abarca diferentes técnicas y algoritmos para el procesamiento de imágenes. A lo largo de los proyectos, se implementan funcionalidades como recorte, aplicación de filtros, dibujo sobre imágenes y edición en tiempo real utilizando la cámara.

## Requisitos

- Python 3.x
- OpenCV
- TensorFlow
- NumPy
- Matplotlib
- Otros paquetes listados en `requirements.txt`

## Instalación

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/tu_usuario/procesamiento_imagenes.git
   cd procesamiento_imagenes
2. Crear un entorno virtual e instalar las dependencias:
   ```bash
    Copiar código
    python -m venv venv
    source venv/bin/activate   # En Windows usa `venv\Scripts\activate`
    pip install -r requirements.txt


   
¡Por supuesto! Aquí tienes el contenido a partir de la sección de instalación:

markdown
Copiar código
## Instalación

1. Clonar el repositorio:
   ```bash
   git clone https://github.com/tu_usuario/procesamiento_imagenes.git
   cd procesamiento_imagenes
2. Crear un entorno virtual e instalar las dependencias:
```bash
python -m venv venv
source venv/bin/activate   # En Windows usa `venv\Scripts\activate`
pip install -r requirements.txt
```

## Uso
Cada proyecto se encuentra en su respectiva carpeta dentro del repositorio. Para ejecutar un proyecto, navega a la carpeta correspondiente y ejecuta el script principal.

Por ejemplo, para ejecutar el proyecto de recorte de imágenes:

```bash
cd recorte_imagenes
python recorte.py
```

## Proyectos
Recorte de Imágenes: Herramienta para recortar partes de una imagen seleccionadas manualmente.
Aplicación de Filtros: Aplicación de diversos filtros (como desenfoque, escala de grises, etc.) a imágenes.
Dibujo sobre Imágenes: Permite dibujar formas y texto sobre imágenes.
Edición en Tiempo Real: Edición de imágenes en tiempo real utilizando la cámara del dispositivo.
Clasificación de Dígitos (MNIST): Clasificador de dígitos escritos a mano utilizando el dataset MNIST y TensorFlow.
Cada carpeta de proyecto contiene más detalles sobre su uso específico y ejemplos de resultados.

## Contribuciones
¡Las contribuciones son bienvenidas! Si deseas contribuir, por favor sigue los siguientes pasos:

Haz un fork del proyecto
Crea una rama (git checkout -b feature/nueva-funcionalidad)
Realiza tus cambios y haz commits (git commit -am 'Añadir nueva funcionalidad')
Haz push a la rama (git push origin feature/nueva-funcionalidad)
Abre un Pull Request
Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo LICENSE para más detalles.
