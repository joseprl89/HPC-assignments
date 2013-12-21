#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>

#define Tolerance 0.00001
#define TRUE 1
#define FALSE 0

#define N 4001
#define ITERATIONS 100

// A matrix that contains all the data for practical purposes
double ** A;

// An array with the linearized content of the previous matrix.
// Based on http://www.cs.arizona.edu/classes/cs522/fall12/examples/mpi-mm.c
// This buffer eases up the gather process.
double* buffer;

void printMatrix(double** A, int n) {
	// To avoid printing too much data.
	if ( n > 5 ) {
		return;
	}

	printf(">>> PRINT MATRIX:\n");
	for ( int i = 0 ; i < n ; i++ ) {
		printf("(");
		for ( int j = 0 ; j < n ; j++ ) {
			printf("%f ", A[i][j]);
		}
		printf(")\n");
	}
}

void printBuffer(double* A, int n) {
	printf(">>> PRINT BUFFER:\n");
	printf("(");
	for ( int j = 0 ; j < n ; j++ ) {
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


printMatrix(A,n);



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
	MPI_Init(NULL,NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &id);
	MPI_Comm_size(MPI_COMM_WORLD, &processes);
	printf("I am %d of %d\n", id, processes);

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


	MPI_Finalize();
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
	int i;
	long t_start,t_end;
	double time;

	// This buffer allocation method forces to hold data in a single dimension vector without gaps.
	buffer = malloc ( (N) * (N) * sizeof (double) );
	A = malloc( (N) * sizeof(double *) );
	for (i=0; i < N ; i++) {
		A[i] = &buffer[i * N];
	}

	initialize(A, N);

	printf("Starting measurement of data with a problem size of %d and %d iterations\n", N, ITERATIONS);
	t_start=usecs();
	solveMPI(A, N);
	t_end=usecs();

	time = ((double)(t_end-t_start))/1000000;
	printf("Computation time = %f\n", time);
	free(buffer);

}
