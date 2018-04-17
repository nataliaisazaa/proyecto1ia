
# coding: utf-8

# # INTELIGENCIA ARTIFICIAL
# 
# ## PROYECTO 6: Clasificador de Plantas.
# 
# ### Integrantes: 
# - Juan David Cardona Molina.
# - Natalia Isaza
# - Santiago Giraldo.
# 
# ### Breve Introducción:
# Este programa esta diseñado para el reconocimiento de plantas mediante captura de vídeo.
# 
# El entrenamiento usado se hizo mediante las herramientas por defecto del OpenCV.
# 
# Opencv_createsamples para el conjunto de imágenes positivas (Imágenes que contenan plantas), este programa creará un archivo de vector.
# 
# Opencv_traincascade el entrenamiento, usando el vector creado por el opencv_createsamples y el conjunto de imágenes negativas, dando como resultado, un archivo .xml de entrenamiento.
# 
# Una vez hecho lo anterior, se carga con el clasificador que trae la librería cv2 de Opencv para python.

# In[10]:


# LIBRERÍAS
import numpy as np # Importamos la librería numpy con alias np.
import cv2 as cv # Importamos la librería cv2 con alias cv.

# CUERPO DEL PROGRAMA
def cargar_entrenamiento():
    entrenamiento = cv.CascadeClassifier('cascade.xml') # Cargamos el entrenamiento.
    return entrenamiento
    
def reconocimiento():    
    entrenamiento = cargar_entrenamiento() # Guardo lo que retorne la función entrenamiento.
    
    captura = cv.VideoCapture(0) # Inicializamos la captura de vídeo.
    # captura = cv.VideoCapture(0) - Para abrir Camara.

    while(True):
        ret, imagen = captura.read() # Leemos lo que hay en la variable captura y lo guardamos en dos variables.
        gris = cv.cvtColor(imagen, cv.COLOR_BGR2GRAY) # Convertimos la imagen a blanco y negro.
        plantas = entrenamiento.detectMultiScale(gris, 1.3, 5) # Buscamos las coordenadas y guardamos su posición.

        for (posicion_x,posicion_y,ancho,largo) in plantas:
            cv.rectangle(imagen,(posicion_x,posicion_y),(posicion_x+ancho,posicion_y+largo),(125,255,0),2) # Crea un rectangulo en las coordenadas.

        cv.imshow('img',imagen) # Mostramos la imagen.

        if cv.waitKey(1) & 0xFF == ord('e'): # Con la tecla E cerramos el programa.
            break

    captura.release()
    cv.destroyAllWindows()

def main():
    reconocimiento() # Llamando la función del reconocimiento de plantas.


# In[9]:


if __name__ == '__main__':
    main()

