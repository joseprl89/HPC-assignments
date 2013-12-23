#!/bin/bash

echo "PROBLEMSIZE, ITERATIONS, PROCESSES, TIME,JOB";
#Iterate over problem size
for problem_size in 200 500 1000 2000;
do
	mpirun --hostfile mpihosts-np 8 bin/solver $problem_size 1000;
done;

# Iterate over iterations.
for iters in 1 10 100 1000 2000;
do
	mpirun --hostfile mpihosts -np 8 bin/solver 1000 $iters;
done;

# Iterate over num threads.
for THREADS in 2 4 8;
do
	mpirun --hostfile mpihosts -np $THREADS bin/solver 1000 1000;
done;