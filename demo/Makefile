CC = g++ -std=c++14
FLAGS = -g -w
ALL = $(CC) $(FLAGS)
datapath = ../processing/*.c*
algopath = ../algorithms/*/*.c*
OBJ = *.cpp $(datapath) $(algopath)
exefile = demo.exe

$(exefile): $(OBJ)
	$(ALL) $^ -o $(exefile)
%.o: %.cpp
	$(ALL) -c $^

cleanw:
	del /Q *.o *.exe
