# Nome do compilador
CC = g++

# Flags de compilação
CFLAGS = -std=c++11 -fopenmp -O2

# Diretórios
INCLUDE_DIR = include
SRC_DIR = srcOpenMP/OpenMP
OBJ_DIR = obj
BIN_DIR = bin

# Arquivos fonte e binário
SOURCES = $(SRC_DIR)/Dataset.cpp $(SRC_DIR)/Network.cpp main.cpp
OBJECTS = $(patsubst %.cpp, $(OBJ_DIR)/%.o, $(notdir $(SOURCES)))
EXEC = $(BIN_DIR)/program

# Regras de compilação
all: $(EXEC)

$(EXEC): $(OBJECTS)
	mkdir -p $(BIN_DIR)
	$(CC) $(CFLAGS) -o $@ $^

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

main.o: main.cpp
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -c main.cpp -o $(OBJ_DIR)/main.o

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

