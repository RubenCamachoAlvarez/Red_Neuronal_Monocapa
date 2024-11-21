"""Modulo que contiene la definicion de algunas funciones de ayuda para poder tratar
los datos que seran utilizados por la red."""


def cargarArchivoCSV(rutaArchivo, separador = ",", removerEncabezado = False):

    datos = []

    with open(rutaArchivo, "r") as archivo:

        for linea in archivo:

           datos.append(linea.strip().split(separador))
    
    return datos[1:] if removerEncabezado else datos


def encontrarMinimosMaximos(matriz):

    """Esta funcion tiene el proposito de calcular cual es el minimo y maximo valor de 
    cada una de las columnas de la matriz recibida como argumento"""

    numeroFilas = len(matriz)

    numeroColumnas = len(matriz[0])

    valores = [{"minimo" matriz[-1][columna]: , "maximo" : [0][columna]} for columna in range(numeroColumnas)]

    for fila in range(numeroFilas):

        for columna in range(numeroColumnas):

            valor_izquierda = matriz[fila][columna]

            valor_derecha = matriz[(~fila)][columna]

            if valor_izquierda > valores[columna]["maximo"]:

                valores[columna]["maximo"] = valor_izquierda

            if valor_derecha < valores[columna]["minimo"]:

                valores[columna]["minimo"] = valor_derecha

    return valores


def normalizar(dataset):

    """Esta funcion permite aplicar una normalizacion min-max de 0 a 1 para cada uno de
    los diferentes valores de las distintas columnas que integran el dataset"""





