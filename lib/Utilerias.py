"""Modulo que contiene la definicion de algunas funciones de ayuda para poder tratar
los datos que seran utilizados por la red."""

import random


def cargarArchivoCSV(rutaArchivo, separador = ",", removerEncabezado = False):

    datos = []

    with open(rutaArchivo, "r") as archivo:

        for linea in archivo:

           datos.append(linea.strip().split(separador))
    
    return datos[1:] if removerEncabezado else datos


def imprimirMatriz(matriz):

    for fila in matriz:

        print(fila)

def matrizEnteros(matriz):

    """Esta funcion convierte una matriz que contiene valores numericos
    representados como cadenas, a su matriz equivalente pero ahora
    almacenando los datos como tipos de dato int."""

    return [[int(elemento)for elemento in fila] for fila in matriz]


def encontrarMinimosMaximos(matriz):

    """Esta funcion tiene el proposito de calcular cual es el minimo y maximo valor de 
    cada una de las columnas de la matriz recibida como argumento"""

    numeroFilas = len(matriz)

    numeroColumnas = len(matriz[0])

    valores = [{"minimo" : matriz[-1][columna] , "maximo" : matriz[0][columna]} for columna in range(numeroColumnas)]

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
    los diferentes valores de las distintas columnas que integran un dataset expresado como matriz"""

    datasetNormalizado = []

    informacionColumnas = encontrarMinimosMaximos(dataset)

    for fila in range(len(dataset)):

        filaNormalizada = []

        for columna in range(len(dataset[0])):

            valor = dataset[fila][columna]

            datosColumna = informacionColumnas[columna]

            filaNormalizada.append((valor - datosColumna["minimo"]) / (datosColumna["maximo"] - datosColumna["minimo"]))

        datasetNormalizado.append(filaNormalizada)

    return datasetNormalizado


def matrizTranspuesta(matriz):

    transpuesta = [[] for columna in range(len(matriz[0]))]

    for fila in matriz:

        for indiceColumna in range(len(matriz[0])):

            transpuesta[indiceColumna].append(fila[indiceColumna])


    return transpuesta


def dividirMatriz(matriz):

    """Esta funcion tiene el proposito de partir una matriz exactamente
    por la mitad. Por ello, el número de columnas de la matriz debe de ser
    un número par."""

    numeroColumnas = len(matriz[0])

    if numeroColumnas % 2 != 0:

        raise ValueError("Esta matriz no tiene un número par de columnas")

    submatrices = ([],[])

    columnaLimite = numeroColumnas // 2

    for fila in matriz:

        submatrices[0].append(fila[:columnaLimite])

        submatrices[1].append(fila[columnaLimite:])

    return submatrices


def dividirDatos(dataset, ratio):

    """Esta función tiene la finalidad de dividir el dataset pasado como
    argumento, en datos para entrenamiento y datos para prueba.
    
    Esta función recibe como argumentos el dataset del que se quieren dividir
    los datos, así como el ratio (un valor entre 0.0 y 1.0) que indique
    el porcentaje de datos que se quieren para entrenamiento.

    Esta función devuelve una tupla, donde el primer elemento contiene la
    porcion del dataset, delimitado por el ratio pasado por argumento, que
    servira para realizar el entrenamiento de la red; mientras que como
    segundo elemento de la tupla contiene el resto del dataset que debería
    de ser utilizado para probar la red una vez que esté entrenada.
    """

    if not (0.0 <= ratio <= 1):

        raise ValueError("El valor del ratio debe de ser entre 0.0 y 1.0")

    datasetPrueba = dataset[:]

    dimensionEntrenamiento = int(len(dataset) * ratio)

    random.shuffle(datasetPrueba)

    datasetEntrenamiento = []

    while len(datasetEntrenamiento) < dimensionEntrenamiento:

        datasetEntrenamiento.append(datasetPrueba.pop())

    return (datasetEntrenamiento, datasetPrueba) 


def calcularRendimientoRed(redNeuronalMonocapa, matrizEntradas, matrizSalidasEsperadas):

    if len(matrizEntradas) != len(matrizSalidasEsperadas):

        raise ValueError("La matriz de entradas y la matriz de salidas de salidas esperadas deben de tener el mismo número de filas")

    casosCorrectos = []

    casosIncorrectos = []
   
    for indiceCaso in range(len(matrizEntradas)):

        vectorEntradas = matrizEntradas[indiceCaso]

        salidasEsperadas = matrizSalidasEsperadas[indiceCaso]

        salidasRed = redNeuronalMonocapa.probar(vectorEntradas)

        registro = (vectorEntradas, salidasEsperadas, salidasRed)

        if salidasRed == salidasEsperadas:

            casosCorrectos.append(registro)
        else:

            casosIncorrectos.append(registro)

    precision = len(casosCorrectos) / len(matrizEntradas)

    return (casosCorrectos, casosIncorrectos, precision)

        
def imprimirCasos(casos):

    print("-"*55)
    
    print("{:15}|{:20}|{:20}".format("Entradas", "Salidas esperadas", "Salidas obtenidas"))

    print("-"*55)

    for registro in casos:

        entradas, salidasEsperadas, salidasObtenidas = registro

        print("{:15}|{:20}|{:20}".format(str(entradas), str(salidasEsperadas), str(salidasObtenidas)))



