from modelo.Perceptron import Perceptron

class RedNeuronalMonocapa:

    """Red neuronal monocapa que se integra de una unica capa de perceptrones en la
    cula se realiza todo el procesamiento para generar la salida solicita"""

    def __init__(self, numeroEntradas, numeroNeuronas, factorAprendizaje):

        self.numeroEntradas = numeroEntradas

        self.neuronas = [Perceptron(numeroEntradas, factorAprendizaje) for i in range(numeroNeuronas)]



    def getNumeroNeuronas(self):

        return len(self.neuronas)



    def entrenar(self, matrizEntradas, matrizSalidas, epoch):

        if len(matrizSalidas) != self.getNumeroNeuronas():

            raise ValueError("El numero de filas de la matriz de salidas no corresponde con el numero de neuronas de la red monocapa")

        while epoch:

            for indice in range(self.getNumeroNeuronas()):

                vectorSalidas = matrizSalidas[indice]

                self.neuronas[indice].entrenar(matrizEntradas, vectorSalidas)

            epoch -= 1

    def probar(self, vectorEntradas):

        salidas = []

        for indice in range(self.getNumeroNeuronas()):

            salida = self.neuronas[indice].computarSalida(vectorEntradas)

            salidas.append(salida)

        return salidas


    def imprimirInformacion(self):

        print("Número de entradas:", self.numeroEntradas)

        print("Número de neuronas:", self.getNumeroNeuronas(), end="\n"*2)

        for indice in range(self.getNumeroNeuronas()):

            print(f"Neurona {indice}: ", end="")

            self.neuronas[indice].imprimirInformacion()
