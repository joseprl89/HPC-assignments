#include <mpi.h>
#include <stdio.h>
#include <sys/time.h>
#include <stdlib.h>
#include <math.h>

#define MICROS_IN_SECOND 1000000

	/**
	 * get microdifference between two timevals.
	 */
	long microsDifference(struct timeval *tv, struct timeval *tv2) {
		long start = (tv->tv_sec * MICROS_IN_SECOND + tv->tv_usec);
		long end = (tv2->tv_sec * MICROS_IN_SECOND + tv2->tv_usec);
		return start - end;
	}

	/**
	 * Sends pings back and forth accross the nodes.
	 */
	int pingData (int dataSize, int repetitions, int tag, int id) {
		// Status
		MPI_Status status;

		char* sent;
		char* received;
		sent = (char*) malloc(dataSize * sizeof(char));
		received = (char*) malloc(dataSize * sizeof(char));

		// Repeat as many times as required.
		for ( int i = 0 ; i < repetitions ; i++ ) {
			// If my id is number 0, send ping then receive pong.
			if (id == 0) {
				MPI_Send(sent, dataSize * sizeof(char), MPI_CHAR, 1, tag, MPI_COMM_WORLD);
				MPI_Recv(received, dataSize * sizeof(char), MPI_CHAR, 1, tag, MPI_COMM_WORLD, &status);
			}
			// If it is 1, pong then ping.
			else {
				MPI_Recv(received, dataSize * sizeof(char), MPI_CHAR, 0, tag, MPI_COMM_WORLD, &status);
				MPI_Send(sent, dataSize * sizeof(char), MPI_CHAR, 0, tag, MPI_COMM_WORLD);
			}
		}

		free(sent);
		free(received);
	}

	/**
	 * Main method
	 */
	int main(int argc, char** argv) {
		struct timeval tStartup, tInit, tFinish,tLoopStart,tLoopEnd;
		gettimeofday(&tInit, NULL);
	
		int id, tag=1;		

		int repetitions = 1000;

		MPI_Init(&argc, &argv);
		MPI_Comm_rank(MPI_COMM_WORLD, &id);

		// CSV header
		if ( id == 0 ) {
			printf("ProcessId, bytesize, repetitions, elapsed\n");
		}

		// Barrier the processes before starting for.
		MPI_Barrier(MPI_COMM_WORLD);
		// Measure time of day.
		gettimeofday(&tStartup, NULL);

		for ( int i = 2 ; i < (2<<20) ; i = i << 1) {
			// Measure time of day.
			gettimeofday(&tLoopStart, NULL);

			// Ping data, 100 repetitions
			pingData (i,repetitions,tag, id);

			// Barrier the processes to ensure all have ended correctly
			MPI_Barrier(MPI_COMM_WORLD);

			// Get time of day again to compare to startup.
			gettimeofday(&tLoopEnd, NULL);

			// Print result in CSV format.
			printf("%d,%d,%d,%lu\n", id, i , repetitions, microsDifference(&tLoopEnd, &tLoopStart));
		}

		MPI_Finalize();
		if ( id == 0 ) {
			// To remove no csv output use grep -v NOCSV
			printf("NOCSV >>>>> Initialized in %lu micros.\n", microsDifference(&tStartup, &tInit));
		}
	}
