CC = g++ -std=c++14
FLAGS = -g -w
# SRC = $(wildcard *.cpp ../*.cpp)
# OBJ = $(patsubst %.cpp, %.o, $(SRC))
exefile = test.exe

$(exefile): knn.o main.o
	$(CC) knn.o main.o -o $(exefile)
knn.o: ../knn.cpp
	$(CC) $(FLAGS) -c ../knn.cpp
main.o: main.cpp
	$(CC) $(FLAGS) -c main.cpp

cleanw:
	del /Q *.o $(exefile)
