import os
import numpy as np
from FileReader import FileReader
from knn import KNN
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
#from tkinter import messagebox
from tkinter import scrolledtext
from tkinter import ttk

# def makeGraphic():

def seleccionar_archivo():
    archivo_path = filedialog.askopenfilename()
    etiqueta_path.config(text=archivo_path)

def guardar_datos_por_indices(data_array, indices_lista, nombre_archivo):
    with open(nombre_archivo, 'w') as file:
        for lista_indices in indices_lista:
            for indice in lista_indices:
                # Obtener los atributos y la etiqueta de clase del arreglo para el índice actual
                atributos = data_array[indice][:-1]  # Excluir el último elemento que es la etiqueta de clase
                etiqueta_clase = int(data_array[indice][-1])  # Convertir la etiqueta de clase a entero

                # Escribir los atributos seguidos de la etiqueta de clase en el archivo
                file.write(','.join(map(str, atributos)) + ',' + str(etiqueta_clase) + '\n')

def probar():
    fileName = etiqueta_path.cget("text")
    k = int(entrada1.get())
    folds = int(entrada2.get())
    valida = option_var.get()

    print("Valida: ")
    print(valida)

    if valida == "Estratificada":
        valType = True
    else:
        valType = False

    #creamos nuestro objeto clasificador
    fr = FileReader(fileName)

    clasificador = KNN(fr.arrInf, folds, k, valType)
    clasificador.make_KNN()

    #print(clasificador)

    # Muestra la información en el área de texto
    texto_info.insert(tk.END, f"Archivo seleccionado: {fileName}\n")
    texto_info.insert(tk.END, f"Valor de KNN: {k}\n")
    texto_info.insert(tk.END, f"Folds: {folds}\n")
    texto_info.insert(tk.END, f"Tipo de Validacion: {valType}\n")
    texto_info.insert(tk.END, f"Resultado de la prueba: ...\n\n")

    #ahora extraemos los datos del clasificador:
    texto_info.insert(tk.END, f"Exactitud Meadia Obtenida: {clasificador.exactitudMedia: .4f}%\n")
    texto_info.insert(tk.END, f"Numero de elementos descartados: {len(clasificador.noisyElements)}\n\n")
    texto_info.insert(tk.END, f"Indice de los elementos descartados:\n")
    texto_info.insert(tk.END, clasificador.noisyElements, "\n")
    texto_info.insert(tk.END, "\n")

    # Colores para cada clase
    colores = ['blue', 'red', 'green', 'orange', 'purple', 'cyan', 'yellow', 'magenta', 'darkgreen', 'pink']
    colores2 = ['#000099', '#cc0000', '#006600', '#cc6600', '#660099', '#006666', '#999900', '#990066', '#003300', '#cc3399']

    # Crear figura y ejes
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    fr.getRelated()
    print(type(fr.related))

    # Graficar los datos originales
    for i, clase in enumerate(fr.arregloDeClase):
        datos_clase = fr.arrInf[fr.arrInf[:, -1] == clase]
        x = datos_clase[:, fr.related[0]]
        y = datos_clase[:, fr.related[1]]
        axs[0].scatter(x, y, color=colores2[i], alpha=0.4, marker='o')

    # Graficar los puntos seleccionados por índices
    for i, sublist in enumerate(clasificador.cleanDataList):
        x_selected = fr.arrInf[sublist, fr.related[0]]
        y_selected = fr.arrInf[sublist, fr.related[1]]
        clases_selected = fr.arrInf[sublist, -1]
        for j, clase in enumerate(clases_selected):
            axs[1].scatter(x_selected[j], y_selected[j], color=colores[int(clase)], alpha=0.5, marker='o')

    # Agregar etiquetas y leyendas
    axs[0].set_xlabel('Característica 1')
    axs[0].set_ylabel('Característica 2')
    axs[0].set_title('Datos Originales')

    axs[1].set_xlabel('Característica 1')
    axs[1].set_ylabel('Característica 2')
    axs[1].set_title('Puntos Seleccionados por Índices')

    plt.tight_layout()
    plt.show()


    # Guardamos Datos
    nombre_archivo = 'clean_data.txt'
    guardar_datos_por_indices(clasificador.dataSet, clasificador.cleanDataList, nombre_archivo)
    print(f"Los datos se han guardado en el archivo '{nombre_archivo}'.")

    #messagebox.showinfo("File name.", fileName)


# Crear la ventana
ventana = tk.Tk()
ventana.title("Proyecto Tratamiento de la Informacion.")
ventana.geometry("1200x600")

# Etiqueta para mostrar el path del archivo seleccionado
etiqueta_path = tk.Label(ventana, text="Seleccione un archivo.")
etiqueta_path.place(x=150, y=20)

# Botón para seleccionar un archivo
boton_seleccionar = tk.Button(ventana, text="Seleccionar Archivo", command=seleccionar_archivo)
boton_seleccionar.place(x=20, y=20)

# Crear etiquetas y campos de entrada
etiqueta1 = tk.Label(ventana, text="Ingrese KNN:")
etiqueta1.place(x=20, y=80)

entrada1 = tk.Entry(ventana)
entrada1.place(x=100, y=80)

etiqueta2 = tk.Label(ventana, text="Folds:")
etiqueta2.place(x=380, y=80)

entrada2 = tk.Entry(ventana)
entrada2.place(x=420, y=80)

etiqueta3 = tk.Label(ventana, text="Tipo de Validacion:")
etiqueta3.place(x=740, y=80)

#
opciones = ["k-fold", "Estratificada"]
option_var = tk.StringVar()
option_var.set(opciones[0])
option_menu = tk.OptionMenu(ventana, option_var, *opciones)
option_menu.place(x=850, y=80)
option_menu.config(bg="white", fg="Black")

# entrada3 = tk.Entry(ventana)
# entrada3.place(x=850, y=80)

#boton para iniciar pruebas
botonPruebas = tk.Button(ventana, text="PROBAR", command=probar)
botonPruebas.place(x=20, y=130)

# Área de texto para mostrar la información
texto_info = scrolledtext.ScrolledText(ventana, width=80, height=20)
texto_info.place(x=280, y=200)

# Ejecutar la interfaz
ventana.mainloop()