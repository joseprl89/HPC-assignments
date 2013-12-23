#!/bin/bash


echo "PROBLEMSIZE, ITERATIONS, PROCESSES, TIME,JOB";
for problem_size in 10 20 50 100 200 500 1000 2000 5000;
do
	for iters in 1 10 100 1000;
	do
		for THREADS in 2 4 8;
		do
			/usr/bin/mpirun -np $THREADS bin/solver $problem_size $iters;
		done;
	done;

done;
