#include <mpi.h>
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>

#define MICROS_IN_SECOND 1000000

// Method based on http://stackoverflow.com/questions/1468596/c-programming-calculate-elapsed-time-in-milliseconds-unix
// To calculate ellapsed time in milliseconds.

/* Return 1 if the difference is negative, otherwise 0.  */
int timeval_subtract(struct timeval *result, struct timeval *t2, struct timeval *t1) {
    long int diff = (t2->tv_usec + 1000000 * t2->tv_sec) - (t1->tv_usec + 1000000 * t1->tv_sec);
    result->tv_sec = diff / 1000000;
    result->tv_usec = diff % 1000000;

    return (diff<0);
}

long millisDifference(struct timeval *tv, struct timeval *tv2) {
	long start = (tv->tv_sec * MICROS_IN_SECOND + tv->tv_usec);
	long end = (tv2->tv_sec * MICROS_IN_SECOND + tv2->tv_usec);
	return start - end;
}

/**
 * Main method
 */
int main(int argc, char** argv) {
	struct timeval tStartup, tInit, tPing, tPong, tFinish;
	gettimeofday(&tInit, NULL);
	
	int MyProc, tag=1;
	char msg='A', msg_recpt;
	MPI_Status status;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &MyProc);
	printf("Process # %d started \n", MyProc);
	
	MPI_Barrier(MPI_COMM_WORLD);
	gettimeofday(&tStartup, NULL);
	
	if (MyProc == 0) {
		gettimeofday(&tPing, NULL);
		printf("Sending message to Proc #1 \n") ;
		MPI_Send(&msg, 1, MPI_CHAR, 1, tag, MPI_COMM_WORLD);

		MPI_Recv(&msg_recpt, 1, MPI_CHAR, 1, tag, MPI_COMM_WORLD, &status);
		printf("Recv'd message from Proc #1 \n") ;
		gettimeofday(&tPong, NULL);
	}
	else {
		MPI_Recv(&msg_recpt, 1, MPI_CHAR, 0, tag, MPI_COMM_WORLD, &status);
		printf("Recv'd message from Proc #0 \n") ;

		printf("Sending message to Proc #0 \n") ;
		MPI_Send(&msg, 1, MPI_CHAR, 0, tag, MPI_COMM_WORLD);
	}

	printf("Finishing proc %d\n", MyProc); 

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();
	gettimeofday(&tFinish, NULL);
	
	
	
	if ( MyProc == 0 ) {
		printf(">>>>>>>>>>>>>> Initialized in %lu micros.\n", millisDifference(&tStartup, &tInit));
		printf(">>>>>>>>>>>>>> Started pinging in %lu micros.\n", millisDifference(&tPing, &tStartup));
		printf(">>>>>>>>>>>>>> Finished ping-pong in %lu micros.\n", millisDifference(&tPong, &tPong));
		printf(">>>>>>>>>>>>>> Finished total execution in %lu micros.\n", millisDifference(&tFinish, &tInit));
	}
}