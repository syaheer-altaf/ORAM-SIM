.PHONY: all exp1 exp2 exp3 clear

exp1: | compiled
	g++ -O2 -std=c++17 ./experiments/experiment_1.cpp -o compiled/experiment_1

exp2: | compiled
	g++ -O2 -std=c++17 ./experiments/experiment_2.cpp -o compiled/experiment_2

exp3: | compiled
	g++ -O2 -std=c++17 ./experiments/experiment_3.cpp -o compiled/experiment_3
compiled:
	mkdir -p compiled

compile: exp1 exp2 exp3

clear:
	rm -rf ./compiled
	rm -rf ./results
