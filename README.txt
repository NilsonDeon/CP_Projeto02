Projeto 01 - Computação Paralela
Grupo 02: Gabriel Vargas, Leticia Americano, Nilson Deon e Olga Camilla
Data: 11/2024

## Referência
O código original não foi desenvolvido pelo grupo.
Ele está disponível no link: https://github.com/alexandremstf/neural-network/tree/master. Autor: Alexandre Magno

## Como executar
Para compilar o código: 
    Codigo Sequencial: make all VERSION=Sequencial
    Codigo OpenMP    : make all VERSION=OpenMP NUM_THREADS=x, sendo x o número de threads desejadas (1, 2, 4 ou 8)
    Codigo MPI       : make all VERSION=MPI NUM_THREADS=x, sendo x o número de threads desejadas (1, 2, 4 ou 8)

Para executar o código: 
    Codigo Sequencial: time ./neuralnetwork
    Codigo OpenMP    : time ./neuralnetwork
    Codigo MPI       : time mpiexec -np y ./neuralnetwork, sendo y o número de processos desejadas (1, 2 ou 4)

## Explicação da aplicação
IRIS.... Perceptron... backpropagation... (explicar)

## Alterações no código

## Resultados
O código foi executado em um computador com as seguintes especificações:

### Versão Sequencial

Tempo: 0m29,424s

### Versão OpenMP

Tempo 1 thread : 0m26,720s
Tempo 2 threads: 0m15,170s
Tempo 4 threads: 0m8,984s
Tempo 8 threads: 0m5,292s

### Versão MPI

Tempo 1 processo  e 4 threads: 0m20,800s
Tempo 2 processos e 2 threads: 0m12,479s
Tempo 4 processo  e 1 thread : 0m9,553s