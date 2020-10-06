#!/bin/bash

g++ -fPIC -c -Wall ../cpp/initializers/he_init.cpp
g++ -shared he_init.o -o libhe_init.so
#gcc -Wl,-rpath ${PWD} -o example example.cpp -L${PWD} -Wall -lpet -lstdc++
