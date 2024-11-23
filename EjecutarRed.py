from lib.Utilerias import *
from modelo.RedMonocapa import RedNeuronalMonocapa

if __name__ == "__main__":

    print("\nMuestra del dataset", end="\n"*2)
    
    muestra = cargarArchivoCSV("datos/Monocapa.csv")

    imprimirMatrizFormato(muestra, encabezado = True)
    
    datos = cargarArchivoCSV("datos/Monocapa.csv", removerEncabezado = True)
    
    dataset = matrizEnteros(datos)
    
    datasetEntrenamiento, datasetPrueba = dividirDatos(dataset, 0.75)
    
    print("Registros de entrenamiento")
    
    imprimirMatrizFormato(datasetEntrenamiento)
    
    print("Registros de prueba")
    
    imprimirMatrizFormato(datasetPrueba)
    
    print("\nNúmero de registros del dataset:", len(dataset))
    
    print("Número de registros de entrenamiento:", len(datasetEntrenamiento))
    
    print("Número de registros de prueba:", len(datasetPrueba), end="\n"*2)
    
    entradasEntrenamiento, salidasEntrenamiento = dividirMatriz(datasetEntrenamiento)
    
    salidasEntrenamiento = matrizTranspuesta(salidasEntrenamiento)
    
    entradasPrueba, salidasEsperadas = dividirMatriz(datasetPrueba)
    
    print("Matriz transpuesta de las salidas para el entrenamiento")

    imprimirMatriz(salidasEntrenamiento)

    
    entradasEntrenamiento = normalizar(entradasEntrenamiento)
    
    entradasPrueba = normalizar(entradasPrueba)
    
    print("\nEntradas normalizadas para el entrenamiento\n")
    
    imprimirMatrizFormato(entradasEntrenamiento)
    
    print("\nEntradas normalizadas para las pruebas\n")
    
    imprimirMatrizFormato(entradasPrueba)
    
    red = RedNeuronalMonocapa(3, 3, 0.4)
    
    red.entrenar(entradasEntrenamiento, salidasEntrenamiento, 100)
    
    casosCorrectos, casosIncorrectos, precisionRed = calcularRendimientoRed(red, entradasPrueba, salidasEsperadas)
    
    print("\nNumero de casos correctos:", len(casosCorrectos))
    
    print("Numero de casos incorrectos:", len(casosIncorrectos))
    
    print(f"Precisión de la red: {precisionRed * 100} %")
    
    print("\nCasos de prueba correctos\n")
    
    imprimirCasos(casosCorrectos)
    
    print("\nCasos de prueba incorrectos\n")
    
    imprimirCasos(casosIncorrectos)
    
    print("\nInformacion de la red\n")
    
    red.imprimirInformacion()
