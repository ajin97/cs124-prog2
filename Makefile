all: strassen

strassen: strassen.o
	g++ strassen.o -o strassen -O3

strassen.o: strassen.cpp
	g++ -Wall -c strassen.cpp -O3
