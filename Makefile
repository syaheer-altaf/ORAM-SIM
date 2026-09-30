.PHONY: exp1 exp2

exp1: | compile
	g++ -O2 -std=c++17 ./experiments/experiment_1.cpp -o compiled/experiment_1

exp2: | compile
	g++ -O2 -std=c++17 ./experiments/experiment_2.cpp -o compiled/experiment_2

compile:
	mkdir -p compiled

clear:
	rm -rf ./compiled
	rm -rf ./results
