.PHONY: all clear test

test: | compiled
	g++ -O2 -std=c++17 ./test/oram_test.cpp -o ./compiled/oram_test

compiled:
	mkdir -p compiled

compile: test

clear:
	rm -rf ./compiled
