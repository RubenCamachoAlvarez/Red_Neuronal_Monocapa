import random
from modelo.FuncionesActivacion import escalon

class Perceptron:

    def __init__(self, numeroEntradas, factorAprendizaje):

        self.pesos = [random.random() for pesoEntrada in range(numeroEntradas)]

        self.factorAprendizaje = factorAprendizaje

        self.umbral = random.random()


    def getNumeroEntradas(self):

        return len(self.pesos)

    
    def computarSalida(self, vectorEntradas):

        numeroEntradas = self.getNumeroEntradas()

        if len(vectorEntradas) != numeroEntradas:

            raise ValueError("El tamaño de vector de entradas no corresponde con el numero de entradas que recibe el perceptron")

        salida = 0.0

        for indice in range(numeroEntradas):

            salida += vectorEntradas[indice] * self.pesos[indice]

        salida -= self.umbral;

        return escalon(salida)


    def entrenar(self, matrizEntradas, vectorSalidas):

        if len(matrizEntradas) != len(vectorSalidas):

            raise ValueError("El numero de filas de la matriz de entradas no corresponde con el numero de elementos del vector que contiene las salidas")

        for indice in range(len(matrizEntradas)):

            vectorEntradas = matrizEntradas[indice]

            salidaEsperada = vectorSalidas[indice]

            salidaComputada = self.computarSalida(vectorEntradas)

            """
            print("Entradas:", vectorEntradas)
            print("Salida Computada:", salidaComputada)
            print("Salida Esperada:", salidaEsperada)

            print("----------------------------------------------")

            """

            if salidaComputada != salidaEsperada:

                self.ajustarPesos(vectorEntradas, salidaEsperada, salidaComputada)


    def ajustarPesos(self, vectorEntradas, salidaEsperada, salidaComputada):

        error = salidaEsperada - salidaComputada

        deltaUmbral = -(self.factorAprendizaje * error)

        self.umbral += deltaUmbral

        for indice in range(self.getNumeroEntradas()):

            deltaPeso = self.factorAprendizaje * error * vectorEntradas[indice]

            self.pesos[indice] += deltaPeso


    def imprimirInformacion(self):

        print("{")

        print("Factor de aprendizaje:", self.factorAprendizaje)

        print("Umbral:", self.umbral, end="\n"*2)

        for indice in range(self.getNumeroEntradas()):

            print(f"Peso {indice}: {self.pesos[indice]}")

        print("}", end="\n"*2)
