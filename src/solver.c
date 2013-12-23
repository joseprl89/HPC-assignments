#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>
#include <omp.h>

#define Tolerance 0.00001
#define TRUE 1
#define FALSE 0

// A matrix that contains all the data for practical purposes
double ** A;

// An array with the linearized content of the previous matrix.
// Based on http://www.cs.arizona.edu/classes/cs522/fall12/examples/mpi-mm.c
// This buffer eases up the gather process.
double* buffer;

int N;
int ITERATIONS;

void printMatrix(double** A, int n) {
	int i,j;
	// To avoid printing too much data.
	if ( n > 5 ) {
		return;
	}

	printf(">>> PRINT MATRIX:\n");
	for ( i = 0 ; i < n ; i++ ) {
		printf("(");
		for ( j = 0 ; j < n ; j++ ) {
			printf("%f ", A[i][j]);
		}
		printf(")\n");
	}
}

void printBuffer(double* A, int n) {
	if ( n > 5 ) {
		return;
	}
	int j;
	printf(">>> PRINT BUFFER:\n");
	printf("(");
	for ( j = 0 ; j < n ; j++ ) {
		printf("%f ", A[j]);
	}
	printf(")\n");
}


/**
 * Initializes data with 1.0 or 0.0
 */
int initialize (double **A, int n) {
   int i,j;

   for (j=0;j<n;j++){
     A[0][j]=1.0;
   }
   for (i=1;i<n;i++){
      A[i][0]=1.0;
      for (j=1;j<n;j++) A[i][j]=0.0;
   }
}

/**
 * Calculates for n iterations the next value.
 */
void solve(double **A, int n) {
   int convergence=FALSE;
   double diff, tmp;
   int i,j, iters=0;
   int for_iters;


   for (for_iters=0;for_iters < ITERATIONS ;for_iters++) 
   { 
     diff = 0.0;
     for (i=1;i<n-1;i++)
     {
       for (j=1;j<n-1;j++)
       {
         tmp = A[i][j];
         A[i][j] = 0.2*(A[i][j] + A[i][j-1] + A[i-1][j] + A[i][j+1] + A[i+1][j]);
         diff += fabs(A[i][j] - tmp);
       }
     }
     iters++;

     if (diff/((double)N*(double)N) < Tolerance)
       convergence=TRUE;
   } /*for*/
}


/**
 * This method has two substantial differences with the previous one:
 * * It is coded using MPI.
 * * It does not modify the previous state of the matrix to calculate C' if the number of processes is larger than 1.
 * The second difference is due to the fact that if we needed to calculate C'[i][j] before C'[i][j+1] (or similars) all the parallelism
 * that could be achieved with this algorithm is almost lost as you would need to perform all operatins serially.
 * 
 * Another limitation known at the moment is that the size of the problem has module number of processes has to be 1, as the allgather method will block the execution until all processes
 * have sent their data. If the problem size is not as previously defined, a process will send 0 bytes, and it will block its peers.
 */
void solveMPI(double** A, int n) {
	int convergence=FALSE;
	double diff, tmp;
	int i,j;
	int for_iters;

	int processes,id;
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &processes);

	// If more processes than rows, finalize.
	if ( id + 1 > n ) {
		MPI_Finalize();
		return;
	}

	
	// Perform iterations
	for ( for_iters = 0 ; for_iters < ITERATIONS ; for_iters++ ) {
		diff = 0.0;
		
		// the processes get a row each.
		for ( i= id + 1 ; i < n-1 ; i+=processes ) {
			for ( j=1 ; j < n-1 ; j++ ) {
				tmp = A[i][j];
				A[i][j] = 0.2 * (A[i][j] + A[i][j-1] + A[i-1][j] + A[i][j+1] + A[i+1][j]);
				diff += fabs(A[i][j] - tmp);
			}
		}

		// Convergence calculation
		if ( diff / ( (double) (N*N) ) < Tolerance) {
			convergence=TRUE;
		}

		if (id == 0) {
			printMatrix(A,n);
		}

		// Gather once every rows / process count.
		for ( i= 1 ; i < n - 1 ; i += processes ) {
			int rowToSend = i+id;
			int sendSize = n;
			if ( rowToSend >= n ) {
//				sendSize = 0;
				break;
			}

			// Gather the result from each process.
			MPI_Allgather(A[rowToSend], sendSize, MPI_DOUBLE, A[i], n, MPI_DOUBLE, MPI_COMM_WORLD);

			//Barrier
			MPI_Barrier(MPI_COMM_WORLD);
		}

		// All should end the iterations at the same pace.
		if (id == 0) {
			printMatrix(A,n);
		}
	}
}


/**
 * This method has two substantial differences with the previous one:
 * * It is coded using MPI.
 * * It does not modify the previous state of the matrix to calculate C' if the number of processes is larger than 1.
 * The second difference is due to the fact that if we needed to calculate C'[i][j] before C'[i][j+1] (or similars) all the parallelism
 * that could be achieved with this algorithm is almost lost as you would need to perform all operatins serially.
 * 
 * Another limitation known at the moment is that the size of the problem has module number of processes has to be 1, as the allgather method will block the execution until all processes
 * have sent their data. If the problem size is not as previously defined, a process will send 0 bytes, and it will block its peers.
 */
void solveMPI2 (double** A, int n) {
	MPI_Status status;
	int convergence=FALSE;
	double diff, tmp;
	int i,j;
	int for_iters;

	int processes,id;
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &processes);

	// Get start and end of the chunk
	int chunkSize = (n-1) / processes;
	int chunkStart = (id * chunkSize);
	int chunkEnd = chunkStart + chunkSize;

	// To assure the first row is not processed
	chunkStart++;
	chunkEnd++;

	// Last process may have overjob.
	if ( id == processes - 1 ) {
		chunkEnd = n - 1;
	}

	// printf("\n\nprocess %d/%d has chunk %d-%d / %d\n\n", id,processes, chunkStart,chunkEnd,chunkSize);

	// Perform iterations
	for ( for_iters = 0 ; for_iters < ITERATIONS ; for_iters++ ) {
		diff = 0.0;
		//printf("%d, Iteration %d\n",id,for_iters);

		// First of all, wait for the previous process to send its calculations if we are not the first one.
		if ( id > 0 && processes > 1 ) {
			// printf("%d, Receiving from %d  at address %d before ran.\n", id, id - 1, chunkStart - 1);
			MPI_Recv(A[chunkStart-1], n, MPI_DOUBLE, id - 1, 0, MPI_COMM_WORLD, &status);
		}

		// the processes get a chunk each until the whole table is processed
		for ( i = chunkStart ; i < chunkEnd ; i ++ ) {
			// printf("%d, Calculating row %d/%d\n",id, i, n);
			for ( j=1 ; j < n-1 ; j++ ) {
				tmp = A[i][j];
				A[i][j] = 0.2 * (A[i][j] + A[i][j-1] + A[i-1][j] + A[i][j+1] + A[i+1][j]);
				diff += fabs(A[i][j] - tmp);
			}
		}

		// Send our calculations to previous process
		if ( id > 0 && processes > 1) {
			// printf("%d, Sending to %d  at address %d\n",id, id - 1, chunkStart);
			MPI_Send(A[ chunkStart ], n, MPI_DOUBLE, id - 1, 0, MPI_COMM_WORLD);
		}

		// If we are not the last one, send and receive from the next one 
		if ( id < processes - 1 && processes > 1 ) {
			// printf("%d, Sending to %d at address %d\n",id, id + 1 , chunkEnd - 1);
			MPI_Send(A[ chunkEnd - 1 ], n, MPI_DOUBLE, id + 1, 0, MPI_COMM_WORLD);
			// printf("%d, Receiving from %d  at address %d\n",id, id + 1 , chunkEnd);
			MPI_Recv(A[ chunkEnd ], n, MPI_DOUBLE, id + 1, 0, MPI_COMM_WORLD, &status);
		}

		// Convergence calculation
		if ( diff / ( (double) (N*N) ) < Tolerance) {
			convergence=TRUE;
		}
	}

	//Barrier execution.
	MPI_Barrier(MPI_COMM_WORLD);
}

/**
 * This method has two substantial differences with the previous one:
 * * It is coded using MPI.
 * * It does not modify the previous state of the matrix to calculate C' if the number of processes is larger than 1.
 * The second difference is due to the fact that if we needed to calculate C'[i][j] before C'[i][j+1] (or similars) all the parallelism
 * that could be achieved with this algorithm is almost lost as you would need to perform all operatins serially.
 * 
 * Another limitation known at the moment is that the size of the problem has module number of processes has to be 1, as the allgather method will block the execution until all processes
 * have sent their data. If the problem size is not as previously defined, a process will send 0 bytes, and it will block its peers.
 */
void solveMPIMP2 (double** A, int n) {
	int nthreads,tid;
	MPI_Status status;
	int convergence=FALSE;
	double diff, tmp;
	int i,j;
	int for_iters;

	int processes,id;
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &processes);

	// Get start and end of the chunk
	int chunkSize = (n-1) / processes;
	int chunkStart = (id * chunkSize);
	int chunkEnd = chunkStart + chunkSize;

	// To assure the first row is not processed
	chunkStart++;
	chunkEnd++;

	// Last process may have overjob.
	if ( id == processes - 1 ) {
		chunkEnd = n - 1;
	}

	// printf("\n\nprocess %d/%d has chunk %d-%d / %d\n\n", id,processes, chunkStart,chunkEnd,chunkSize);

	// Perform iterations
	for ( for_iters = 0 ; for_iters < ITERATIONS ; for_iters++ ) {
		diff = 0.0;
		//printf("%d, Iteration %d\n",id,for_iters);

		// First of all, wait for the previous process to send its calculations if we are not the first one.
		if ( id > 0 && processes > 1 ) {
			// printf("%d, Receiving from %d  at address %d before ran.\n", id, id - 1, chunkStart - 1);
			MPI_Recv(A[chunkStart-1], n, MPI_DOUBLE, id - 1, 0, MPI_COMM_WORLD, &status);
		}

		// the processes get a chunk each until the whole table is processed
		#pragma omp parallel for schedule(static,1) private(i,nthreads,tid)
		for ( i = chunkStart ; i < chunkEnd ; i ++ ) {
			// printf("%d, Calculating row %d/%d\n",id, i, n);
			for ( j=1 ; j < n-1 ; j++ ) {
				tmp = A[i][j];
				A[i][j] = 0.2 * (A[i][j] + A[i][j-1] + A[i-1][j] + A[i][j+1] + A[i+1][j]);
				diff += fabs(A[i][j] - tmp);
			}
		}

		// Send our calculations to previous process
		if ( id > 0 && processes > 1) {
			// printf("%d, Sending to %d  at address %d\n",id, id - 1, chunkStart);
			MPI_Send(A[ chunkStart ], n, MPI_DOUBLE, id - 1, 0, MPI_COMM_WORLD);
		}

		// If we are not the last one, send and receive from the next one 
		if ( id < processes - 1 && processes > 1 ) {
			// printf("%d, Sending to %d at address %d\n",id, id + 1 , chunkEnd - 1);
			MPI_Send(A[ chunkEnd - 1 ], n, MPI_DOUBLE, id + 1, 0, MPI_COMM_WORLD);
			// printf("%d, Receiving from %d  at address %d\n",id, id + 1 , chunkEnd);
			MPI_Recv(A[ chunkEnd ], n, MPI_DOUBLE, id + 1, 0, MPI_COMM_WORLD, &status);
		}

		// Convergence calculation
		if ( diff / ( (double) (N*N) ) < Tolerance) {
			convergence=TRUE;
		}
	}

	//Barrier execution.
	MPI_Barrier(MPI_COMM_WORLD);
}


void solveMPIMP(double** A, int n) {
	/* Fork a team of threads giving them their own copies of variables */
	int nthreads,tid;
	int convergence=FALSE;
	double diff, tmp;
	int i,j;
	int for_iters;

	int processes,id;
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &processes);

	// If more processes than rows, finalize.
	if ( id + 1 > n ) {
		MPI_Finalize();
		return;
	}

	
	// Perform iterations
	for ( for_iters = 0 ; for_iters < ITERATIONS ; for_iters++ ) {
		diff = 0.0;
		
		// the processes get a row each.
		#pragma omp parallel for schedule(static,1) private(i,nthreads,tid)
		for ( i= id + 1 ; i < n-1 ; i+=processes ) {
//			printf("Calculating row %d at process %d thread %d\n", i, id, omp_get_thread_num());
			for ( j=1 ; j < n-1 ; j++ ) {
				tmp = A[i][j];
				A[i][j] = 0.2 * (A[i][j] + A[i][j-1] + A[i-1][j] + A[i][j+1] + A[i+1][j]);
				diff += fabs(A[i][j] - tmp);
			}
		}

		// Convergence calculation
		if ( diff / ( (double) (N*N) ) < Tolerance) {
			convergence=TRUE;
		}

		if (id == 0) {
			//printMatrix(A,n);
		}

		// Gather once every rows / process count.
		for ( i= 1 ; i < n - 1 ; i += processes ) {
			int rowToSend = i+id;
			int sendSize = n;
			if ( rowToSend >= n ) {
//				sendSize = 0;
				break;
			}

			// Gather the result from each process.
			MPI_Allgather(A[rowToSend], sendSize, MPI_DOUBLE, A[i], n, MPI_DOUBLE, MPI_COMM_WORLD);

			//Barrier
			MPI_Barrier(MPI_COMM_WORLD);
		}

		// All should end the iterations at the same pace.
		if (id == 0) {
			//printMatrix(A,n);
		}
	}
}

/**
 * Measures milliseconds elapsed.
 */
long usecs (void) {
  struct timeval t;

  gettimeofday(&t,NULL);
  return t.tv_sec*1000000+t.tv_usec;
}


/**
 * Starts procedure.
 */
int main(int argc, char * argv[]) {
	if ( argc < 3 ) {
		printf("Usage: solver PROBLEM_SIZE ITERATIONS\n");
		return;
	}
	
	N = atoi (argv[1]);
	ITERATIONS = atoi (argv[2]);

	int i,id,processes;
	long t_start,t_end;
	double time;

	// Start mpi (can only be done once without crashing).
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &processes);
	
	// This buffer allocation method forces to hold data in a single dimension vector without gaps.
	buffer = malloc ( (N) * (N) * sizeof (double) );
	A = malloc( (N) * sizeof(double *) );
	for (i=0; i < N ; i++) {
		A[i] = &buffer[i * N];
	}

	// Linear code.
	if ( id == 0 ) {
		initialize(A, N);
		
		t_start=usecs();
		solve(A, N);
		t_end=usecs();

		time = ((double)(t_end-t_start))/1000000;
		printf("%d,%d,%d,%f,LINEAR\n", N, ITERATIONS, processes, time);
		printMatrix(A,N);
	}

	MPI_Barrier(MPI_COMM_WORLD);

	// MPI Code
	initialize(A, N);

	MPI_Barrier(MPI_COMM_WORLD);
	t_start=usecs();
	solveMPI2(A, N);
	t_end=usecs();

	time = ((double)(t_end-t_start))/1000000;
	if ( id == 0) printf("%d,%d,%d,%f,MPI\n", N, ITERATIONS, processes, time);

	// All should end the iterations at the same pace.
	if (id == 0) {
		printMatrix(A,N);
	}
	
	MPI_Barrier(MPI_COMM_WORLD);

	// MPI + MP code.
	initialize(A, N);
	t_start=usecs();
	solveMPIMP2(A, N);
	t_end=usecs();
	time = ((double)(t_end-t_start))/1000000;
	// Output
	if (id == 0) {
		printf("%d,%d,%d,%f,MPI+MP\n", N, ITERATIONS, processes, time);
		printMatrix(A,N);
	}

	// Finalize mpi and buffer.
	MPI_Finalize();
	free(buffer);

}
