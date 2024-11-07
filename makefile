# Compila a versão sequencial por padrao
VERSION = Sequencial

ifeq ($(VERSION),OpenMP)
	COMPILER = g++ -std=c++11 -Wall -g -fopenmp
	SRC_DIR = srcOpenMP
else ifeq ($(VERSION),MPI)
	COMPILER = mpic++ -std=c++11 -Wall -g
	SRC_DIR = srcMPI
else
	COMPILER = g++ -std=c++11 -Wall -g
	SRC_DIR = srcSequencial
endif

EXEC_PROG = neuralnetwork
BINARIES = $(EXEC_PROG)

SOURCES := $(shell find $(SRC_DIR) -name '*.cpp')
OBJECTS = main.o $(SOURCES:.cpp=.o)

all: clean $(EXEC_PROG)
	@echo Neural Network Build Completed

%.o: %.cpp
	$(COMPILER) -c -o $@ $< -w

$(EXEC_PROG): $(OBJECTS)
	$(COMPILER) -o $(EXEC_PROG) $(OBJECTS) 

# prevents make from getting confused
.PHONY : run
run:
	./$(EXEC_PROG)

.PHONY : clean 
clean:
	rm -rf $(EXEC_PROG) $(shell find . -name '*.o')