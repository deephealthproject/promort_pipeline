#! /usr/bin/python3
import psutil
cores = psutil.cpu_count(logical=False)
print(cores//2)
