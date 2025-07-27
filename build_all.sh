gcc ./src/CBOW.c -o cbow -lm -pthread -O3 -march=native -funroll-loops
gcc ./src/CBOW_NS.c -o cbowns -lm -pthread -O3 -march=native -funroll-loops
gcc ./src/Skipgram.c -o skipgram -lm -pthread -O3 -march=native -funroll-loops
gcc ./src/Skipgram_NS.c -o skipgramns -lm -pthread -O3 -march=native -funroll-loops
