.PHONY: all exp1 exp2 clear

exp1: | compiled
	g++ -O2 -std=c++17 ./experiments/experiment_1.cpp -o compiled/experiment_1

exp2: | compiled
	g++ -O2 -std=c++17 ./experiments/experiment_2.cpp -o compiled/experiment_2

compiled:
	mkdir -p compiled

compile: exp1 exp2

clear:
	rm -rf ./compiled
	rm -rf ./results
