# Compila a versão sequencial por padrao
VERSION = Sequencial
COMPILER = mpic++ -std=c++11 -Wall -g


ifeq ($(VERSION),OpenMP)
	SRC_DIR = srcOpenMP
else ifeq ($(VERSION),MPI)
	SRC_DIR = srcMPI
else
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