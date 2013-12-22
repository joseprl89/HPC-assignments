#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

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
		int i;
		// Status
		MPI_Status status;

		char* sent;
		char* received;
		sent = (char*) malloc(dataSize * sizeof(char));
		received = (char*) malloc(dataSize * sizeof(char));

		// Repeat as many times as required.
		for ( i = 0 ; i < repetitions ; i++ ) {
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

	int calculations ( int dataSize, int repetitions ) {
		int i,j;
		float* data = malloc(dataSize * sizeof(float));

		for ( i = 0 ; i < repetitions ; i++ ) {
			for ( j = 0 ; j < dataSize ; j++ ) {
				// Perform some calculations in here.
				float prev = data[j];
				float a = prev * 4; // 1 op
				float b = prev - 3.0f; // 2 op
				float c = prev / 3; // 3 op
				float d = prev + 1.0f; // 4 op
				float e = a + b + c + d; // 5 - 7 op
				data[j] = e / 6.0f; // 8 op
			}
		}
		return dataSize * repetitions * 8;
	}

	/**
	 * Main method
	 */
	int main(int argc, char** argv) {
		int i;
		struct timeval tStartup, tInit, tFinish,tLoopStart,tLoopEnd,tCalcStart,tCalcEnd;
		gettimeofday(&tInit, NULL);
	
		int id, tag=1;		

		int repetitions = 1000;

		MPI_Init(&argc, &argv);
		MPI_Comm_rank(MPI_COMM_WORLD, &id);

		// CSV header
		if ( id == 0 ) {
			printf("ProcessId, problem size, elapsed, process\n");
		}

		// Barrier the processes before starting for.
		MPI_Barrier(MPI_COMM_WORLD);

		// Measure time of day.
		gettimeofday(&tStartup, NULL);

		// Print initialize time.
		printf("%d,0,%lu,INITIALIZE\n", id, microsDifference(&tStartup, &tInit));

		// Measure time of day.
		gettimeofday(&tCalcStart, NULL);

		// Perform 1000 calculations over 1000 elements.
		long flop = calculations(1000, 100000);

		// Barrier the processes before starting for.
		MPI_Barrier(MPI_COMM_WORLD);

		// Measure time of day.
		gettimeofday(&tCalcEnd, NULL);
		

		// Print result in CSV format.
		printf("%d,%lu,%lu,CALCULATIONS\n", id, flop , microsDifference(&tCalcEnd, &tCalcStart) / repetitions);

		// Barrier the processes before starting for.
		MPI_Barrier(MPI_COMM_WORLD);

		for ( i = 2 ; i < (2<<20) ; i = i << 1) {
			// Measure time of day.
			gettimeofday(&tLoopStart, NULL);

			// Ping data, 100 repetitions
			pingData (i,repetitions,tag, id);

			// Barrier the processes to ensure all have ended correctly
			MPI_Barrier(MPI_COMM_WORLD);

			// Get time of day again to compare to startup.
			gettimeofday(&tLoopEnd, NULL);

			// Print result in CSV format.
			printf("%d,%d,%lu,PING\n", id, i , microsDifference(&tLoopEnd, &tLoopStart) / repetitions);
		}

		// Barrier the processes before starting for.
		MPI_Barrier(MPI_COMM_WORLD);
		
		for ( i = 0 ; i < 1000 ; i++ ) {
			// Measure time of day.
			gettimeofday(&tLoopStart, NULL);

			// Ping data, 100 repetitions
			pingData (1024*1024 * 32, 1,tag, id); // 32Mb once.

			// Barrier the processes to ensure all have ended correctly
			MPI_Barrier(MPI_COMM_WORLD);

			// Get time of day again to compare to startup.
			gettimeofday(&tLoopEnd, NULL);

			// Print result in CSV format.
			printf("%d,%d,%lu,PING DEVIATION\n", id, i , microsDifference(&tLoopEnd, &tLoopStart) / repetitions);
		}

		MPI_Finalize();
	}
