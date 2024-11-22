from lib.Utilerias import *
from modelo.RedMonocapa import RedNeuronalMonocapa

datos = cargarArchivoCSV("datos/Monocapa.csv", removerEncabezado = True)

dataset = matrizEnteros(datos)

datasetEntrenamiento, datasetPrueba = dividirDatos(dataset, 0.75)

print("\nNúmero de registros del dataset:", len(dataset))

print("Número de registros de entrenamiento:", len(datasetEntrenamiento))

print("Número de registros de prueba:", len(datasetPrueba), end="\n"*2)

entradasEntrenamiento, salidasEntrenamiento = dividirMatriz(datasetEntrenamiento)

salidasEntrenamiento = matrizTranspuesta(salidasEntrenamiento)

entradasPrueba, salidasEsperadas = dividirMatriz(datasetPrueba)



entradasEntrenamiento = normalizar(entradasEntrenamiento)

entradasPrueba = normalizar(entradasPrueba)



red = RedNeuronalMonocapa(3, 3, 0.2)

red.entrenar(entradasEntrenamiento, salidasEntrenamiento, 100)

casosCorrectos, casosIncorrectos, precisionRed = calcularRendimientoRed(red, entradasPrueba, salidasEsperadas)

print("Numero de casos correctos:", len(casosCorrectos))

print("Numero de casos incorrectos:", len(casosIncorrectos))

print(f"Precisión de la red: {precisionRed * 100} %")

print("\nCasos de prueba correctos\n")

imprimirCasos(casosCorrectos)

print("\nCasos de prueba incorrectos\n")

imprimirCasos(casosIncorrectos)

print("\nInformacion de la red\n")

red.imprimirInformacion()
