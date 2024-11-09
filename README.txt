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

(feito pelo GPT) -> ARRUMAR:

O código é uma implementação de uma rede neural para resolver o problema clássico de classificação do conjunto de dados Iris. O conjunto de dados Iris é um conjunto de dados multivariados introduzido pelo estatístico e biólogo britânico Ronald Fisher em seu artigo de 1936 "The use of multiple measurements in taxonomic problems". É talvez o melhor conhecido banco de dados a ser encontrado na literatura de reconhecimento de padrões.

O conjunto de dados contém 150 instâncias, onde cada instância é uma amostra de medidas das características de uma flor de íris. Cada amostra contém quatro atributos (comprimento e largura das sépalas e pétalas) e uma classe, que é o tipo específico de íris (setosa, versicolor ou virginica).

A rede neural usa o algoritmo de backpropagation para aprender a classificar corretamente as flores de íris com base em suas medidas. A rede é treinada para atingir uma taxa de acerto desejada de 95% com uma tolerância máxima de erro de 0.05. O treinamento é realizado por até 1000 épocas, e a rede pode ter até 15 camadas escondidas. A taxa de aprendizado é definida como 0.25.

Durante o treinamento, a rede ajusta seus pesos para minimizar o erro entre a saída prevista pela rede e a saída real (ou seja, a classe correta da flor de íris). Se a rede atinge a taxa de acerto desejada antes do número máximo de épocas, o treinamento é interrompido. Caso contrário, o treinamento continua até o número máximo de épocas.

Após o treinamento, a rede é capaz de classificar corretamente novas amostras de flores de íris com uma alta taxa de acerto.

## Alterações no código

A função principal do código é o void Network::autoTraining(int hidden_layer_limit, double learning_rate_increase){}
COMPLETAR!!!!

## Resultados
O código foi executado em um computador com as seguintes especificações:

Arquitetura do Sistema: x86_64 (64 bits)
Sistema Operacional: Ubuntu 22.04.1
Versão do Kernel do Linux: 6.8.0-48-generic
Processador: Intel(R) Core(TM) i7-10700F CPU @ 2.90GHz
Núcleos por Soquete: 8
Threads por Núcleo: 2
Soquetes: 1
Total de CPUs (Núcleos * Threads por Núcleo * Soquetes): 16
Memória RAM Total: 15 GiB
Swap Total: 2,0 GiB

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