/*
 * Exact genetic sequence alignment
 * (Using brute force)
 *
 * CUDA version
 *
 * Computacion Paralela, Grado en Informatica (Universidad de Valladolid)
 * 2023/2024
 *
 * v1.3
 *
 * (c) 2024, Arturo Gonzalez-Escribano
 */
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<limits.h>
#include<sys/time.h>

/* Headers for the CUDA assignment versions */
#include<cuda.h>

/* Example of macros for error checking in CUDA */
#define CUDA_CHECK_FUNCTION( call )	{ cudaError_t check = call; if ( check != cudaSuccess ) fprintf(stderr, "CUDA Error in line: %d, %s\n", __LINE__, cudaGetErrorString(check) ); }
#define CUDA_CHECK_KERNEL( )	{ cudaError_t check = cudaGetLastError(); if ( check != cudaSuccess ) fprintf(stderr, "CUDA Kernel Error in line: %d, %s\n", __LINE__, cudaGetErrorString(check) ); }

/* Arbitrary value to indicate that no matches are found */
#define	NOT_FOUND	-1

/* Arbitrary value to restrict the checksums period */
#define CHECKSUM_MAX	65535


/* 
 * Utils: Function to get wall time
 */
double cp_Wtime(){
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + 1.0e-6 * tv.tv_usec;
}

/*
 * Utils: Random generator
 */
#include "rng.c"

/*
 *
 * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
 * DO NOT USE OpenMP IN YOUR CODE
 *
 */
 // TODO INIZIO 
/* ADD KERNELS AND OTHER FUNCTIONS HERE */

/* 
 * KERNEL principale: scandisce la sequenza (a chunk) e controlla se il pattern corrisponde
 *
 * d_sequence:    sequenza globale su GPU
 * seq_length:    lunghezza della sequenza (intera)
 * seqStart:      punto di inizio del chunk da analizzare
 * chunkLen:      lunghezza del chunk in questa invocazione
 *
 * d_patterns:    array di puntatori ai pattern su GPU
 * d_pat_length:  array con le lunghezze di ciascun pattern
 * patStart e patEnd: intervallo dei pattern da processare in questo batch
 *
 * d_pat_found:   array (grande pat_number) dove salviamo la posizione di match 
 *                (inizialmente impostato a ULLONG_MAX se "non trovato").
 * d_pat_matches: contatore globale del numero di pattern trovati
 *
 * La griglia è configurata con dimensioni (gridX, gridY), dove:
 *    - offset (thread x) è l’offset sulla sequenza
 *    - localPat (thread y) è l’indice di pattern relativo nel batch 
 */
__global__ void matchPatternsKernel(
	char* d_sequence, unsigned long seq_length, unsigned long seqStart, unsigned long chunkLen,
	char** d_patterns, unsigned long* d_pat_length, unsigned long patStart, unsigned long patEnd,
	unsigned long long* d_pat_found, int* d_pat_matches)
{

	// Posizione della sequenza che stiamo analizzando
	unsigned long offset = blockIdx.x * blockDim.x + threadIdx.x;
	// Quale pattern stiamo controllando
	unsigned long localPat = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned long patIndex = patStart + localPat; // Indice assoluto del pattern
	unsigned long patLen = d_pat_length[patIndex]; // Lunghezza del pattern

	// Controllo di non andare oltre la lunghezza totale della sequenza
	if (seqStart + offset + patLen > seq_length) {
		return;
	}

	// Controllo di non sforare il chunk
	if (offset + patLen > chunkLen) {
		return;
	}

	// Confronto carattere per carattere
	bool match = true;
	for (unsigned long j = 0; j < patLen; j++) {
		// Se un carattere non corrisponde, il pattern non è presente in questa posizione
		if (d_sequence[seqStart + offset + j] != d_patterns[patIndex][j]) {
			match = false;
			break;
		}
	}

	__syncthreads();

	/*Se c'è corrispondenza: 
	 * - atomicCAS su d_pat_found, in modo che solo il primo thread che entra scrive la posizione (seqStart+offset)
	 * - atomicAdd su d_pat_matches per incrementare il contatore di pattern trovati
	 */
	if (match) {
		if (atomicCAS(&d_pat_found[patIndex], ULLONG_MAX, seqStart + offset) == ULLONG_MAX) {
			atomicAdd(d_pat_matches, 1);
		} else {
			atomicMin(&d_pat_found[patIndex], seqStart + offset);
		}
	}

}

/* 
 * KERNEL per incrementare i match sulla sequenza
 *	un thread per ciascun (patternId, offsetPattern)
 *
 * d_pat_found:   array con tutte le posizioni iniziali trovate (ULLONG_MAX se non trovato match)
 * d_pat_length: lunghezza di ciascun pattern
 * d_seq_matches: array di match sulla sequenza, quante volte ogni posizione della sequenza è stata coperta da uno o più pattern
 * pat_number: totale pattern
 * seq_length:    lunghezza della sequenza (intera)
 */
 __global__ void incrementMatchesKernel(
	const unsigned long long* d_pat_found,
	const unsigned long* d_pat_length,      
	int* d_seq_matches,                     
	int pat_number,
	unsigned long seq_length
) {
	// Quale pattern stiamo processando
	int patId = blockIdx.x * blockDim.x + threadIdx.x;
	// Offset del pattern (posizione della lettera all'interno del pattern)
	int localOffset = blockIdx.y * blockDim.y + threadIdx.y;

	// Fuori dal numero totale di pattern?
	if (patId >= pat_number) return;

	// Posizione iniziale pattern
	unsigned long long startPos = d_pat_found[patId];
	if (startPos == ULLONG_MAX) {
		// Pattern non trovato	
		return;
	}
	// Lunghezza del pattern
	unsigned long length = d_pat_length[patId];
	// Troppo grande? esco
	if (localOffset >= length) return;

	// Calcola la posizione esatta sulla sequenza
	unsigned long long pos = startPos + localOffset;
	// Controllo per non sforare la sequenza
	if (pos >= seq_length) return;

	// Incremento atomico del numero di match nella sequenza 
	atomicAdd(&d_seq_matches[pos], 1);
}


/*
 * Data una seq_length, restituisce la dimensione del chunk
 * da usare per la suddivisione della sequenza.
 */
unsigned long get_chunk_size(unsigned long seq_length) {
	if (seq_length < 1024) return seq_length;  // Se piccolo, nessuna suddivisione
	if (seq_length < 1000000) return max(1024UL, min(65536UL, (unsigned long)seq_length / 8));
	return max(65536UL, min(1048576UL, (unsigned long)seq_length / 10));
}

// TODO FINE
/*
 *
 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
 *
 */

/*
 * Function: Allocate new patttern
 */
char *pattern_allocate( rng_t *random, unsigned long pat_rng_length_mean, unsigned long pat_rng_length_dev, unsigned long seq_length, unsigned long *new_length ) {

	/* Random length */
	unsigned long length = (unsigned long)rng_next_normal( random, (double)pat_rng_length_mean, (double)pat_rng_length_dev );
	if ( length > seq_length ) length = seq_length;
	if ( length <= 0 ) length = 1;

	/* Allocate pattern */
	char *pattern = (char *)malloc( sizeof(char) * length );
	if ( pattern == NULL ) {
		fprintf(stderr,"\n-- Error allocating a pattern of size: %lu\n", length );
		exit( EXIT_FAILURE );
	}

	/* Return results */
	*new_length = length;
	return pattern;
}

/*
 * Function: Fill random sequence or pattern
 */
void generate_rng_sequence( rng_t *random, float prob_G, float prob_C, float prob_A, char *seq, unsigned long length) {
	unsigned long ind;
	for( ind=0; ind<length; ind++ ) {
		double prob = rng_next( random );
		if( prob < prob_G ) seq[ind] = 'G';
		else if( prob < prob_C ) seq[ind] = 'C';
		else if( prob < prob_A ) seq[ind] = 'A';
		else seq[ind] = 'T';
	}
}

/*
 * Function: Copy a sample of the sequence
 */
void copy_sample_sequence( rng_t *random, char *sequence, unsigned long seq_length, unsigned long pat_samp_loc_mean, unsigned long pat_samp_loc_dev, char *pattern, unsigned long length) {
	/* Choose location */
	unsigned long  location = (unsigned long)rng_next_normal( random, (double)pat_samp_loc_mean, (double)pat_samp_loc_dev );
	if ( location > seq_length - length ) location = seq_length - length;
	if ( location <= 0 ) location = 0;

	/* Copy sample */
	unsigned long ind;
	for( ind=0; ind<length; ind++ )
		pattern[ind] = sequence[ind+location];
}

/*
 * Function: Regenerate a sample of the sequence
 */
void generate_sample_sequence( rng_t *random, rng_t random_seq, float prob_G, float prob_C, float prob_A, unsigned long seq_length, unsigned long pat_samp_loc_mean, unsigned long pat_samp_loc_dev, char *pattern, unsigned long length ) {
	/* Choose location */
	unsigned long  location = (unsigned long)rng_next_normal( random, (double)pat_samp_loc_mean, (double)pat_samp_loc_dev );
	if ( location > seq_length - length ) location = seq_length - length;
	if ( location <= 0 ) location = 0;

	/* Regenerate sample */
	rng_t local_random = random_seq;
	rng_skip( &local_random, location );
	generate_rng_sequence( &local_random, prob_G, prob_C, prob_A, pattern, length);
}


/*
 * Function: Print usage line in stderr
 */
void show_usage( char *program_name ) {
	fprintf(stderr,"Usage: %s ", program_name );
	fprintf(stderr,"<seq_length> <prob_G> <prob_C> <prob_A> <pat_rng_num> <pat_rng_length_mean> <pat_rng_length_dev> <pat_samples_num> <pat_samp_length_mean> <pat_samp_length_dev> <pat_samp_loc_mean> <pat_samp_loc_dev> <pat_samp_mix:B[efore]|A[fter]|M[ixed]> <long_seed>\n");
	fprintf(stderr,"\n");
}



/*
 * MAIN PROGRAM
 */
int main(int argc, char *argv[]) {
	/* 0. Default output and error without buffering, forces to write immediately */
	setbuf(stdout, NULL);
	setbuf(stderr, NULL);

	/* 1. Read scenary arguments */
	/* 1.1. Check minimum number of arguments */
	if (argc < 15) {
		fprintf(stderr, "\n-- Error: Not enough arguments when reading configuration from the command line\n\n");
		show_usage( argv[0] );
		exit( EXIT_FAILURE );
	}

	/* 1.2. Read argument values */
	unsigned long seq_length = atol( argv[1] );
	float prob_G = atof( argv[2] );
	float prob_C = atof( argv[3] );
	float prob_A = atof( argv[4] );
	if ( prob_G + prob_C + prob_A > 1 ) {
		fprintf(stderr, "\n-- Error: The sum of G,C,A,T nucleotid probabilities cannot be higher than 1\n\n");
		show_usage( argv[0] );
		exit( EXIT_FAILURE );
	}
	prob_C += prob_G;
	prob_A += prob_C;

	int pat_rng_num = atoi( argv[5] );
	unsigned long pat_rng_length_mean = atol( argv[6] );
	unsigned long pat_rng_length_dev = atol( argv[7] );

	int pat_samp_num = atoi( argv[8] );
	unsigned long pat_samp_length_mean = atol( argv[9] );
	unsigned long pat_samp_length_dev = atol( argv[10] );
	unsigned long pat_samp_loc_mean = atol( argv[11] );
	unsigned long pat_samp_loc_dev = atol( argv[12] );

	char pat_samp_mix = argv[13][0];
	if ( pat_samp_mix != 'B' && pat_samp_mix != 'A' && pat_samp_mix != 'M' ) {
		fprintf(stderr, "\n-- Error: Incorrect first character of pat_samp_mix: %c\n\n", pat_samp_mix);
		show_usage( argv[0] );
		exit( EXIT_FAILURE );
	}

	unsigned long seed = atol( argv[14] );

#ifdef DEBUG
	/* DEBUG: Print arguments */
	printf("\nArguments: seq_length=%lu\n", seq_length );
	printf("Arguments: Accumulated probabilitiy G=%f, C=%f, A=%f, T=1\n", prob_G, prob_C, prob_A );
	printf("Arguments: Random patterns number=%d, length_mean=%lu, length_dev=%lu\n", pat_rng_num, pat_rng_length_mean, pat_rng_length_dev );
	printf("Arguments: Sample patterns number=%d, length_mean=%lu, length_dev=%lu, loc_mean=%lu, loc_dev=%lu\n", pat_samp_num, pat_samp_length_mean, pat_samp_length_dev, pat_samp_loc_mean, pat_samp_loc_dev );
	printf("Arguments: Type of mix: %c, Random seed: %lu\n", pat_samp_mix, seed );
	printf("\n");
#endif // DEBUG

		CUDA_CHECK_FUNCTION( cudaSetDevice(0) );

	/* 2. Initialize data structures */
	/* 2.1. Skip allocate and fill sequence */
	rng_t random = rng_new( seed );
	rng_skip( &random, seq_length );

	/* 2.2. Allocate and fill patterns */
	/* 2.2.1 Allocate main structures */
	int pat_number = pat_rng_num + pat_samp_num;
	unsigned long *pat_length = (unsigned long *)malloc( sizeof(unsigned long) * pat_number );
	char **pattern = (char **)malloc( sizeof(char*) * pat_number );
	if ( pattern == NULL || pat_length == NULL ) {
		fprintf(stderr,"\n-- Error allocating the basic patterns structures for size: %d\n", pat_number );
		exit( EXIT_FAILURE );
	}

	/* 2.2.2 Allocate and initialize ancillary structure for pattern types */
	int ind;
	unsigned long lind;
	#define PAT_TYPE_NONE	0
	#define PAT_TYPE_RNG	1
	#define PAT_TYPE_SAMP	2
	char *pat_type = (char *)malloc( sizeof(char) * pat_number );
	if ( pat_type == NULL ) {
		fprintf(stderr,"\n-- Error allocating ancillary structure for pattern of size: %d\n", pat_number );
		exit( EXIT_FAILURE );
	}
	for( ind=0; ind<pat_number; ind++ ) pat_type[ind] = PAT_TYPE_NONE;

	/* 2.2.3 Fill up pattern types using the chosen mode */
	switch( pat_samp_mix ) {
	case 'A':
		for( ind=0; ind<pat_rng_num; ind++ ) pat_type[ind] = PAT_TYPE_RNG;
		for( ; ind<pat_number; ind++ ) pat_type[ind] = PAT_TYPE_SAMP;
		break;
	case 'B':
		for( ind=0; ind<pat_samp_num; ind++ ) pat_type[ind] = PAT_TYPE_SAMP;
		for( ; ind<pat_number; ind++ ) pat_type[ind] = PAT_TYPE_RNG;
		break;
	default:
		if ( pat_rng_num == 0 ) {
			for( ind=0; ind<pat_number; ind++ ) pat_type[ind] = PAT_TYPE_SAMP;
		}
		else if ( pat_samp_num == 0 ) {
			for( ind=0; ind<pat_number; ind++ ) pat_type[ind] = PAT_TYPE_RNG;
		}
		else if ( pat_rng_num < pat_samp_num ) {
			int interval = pat_number / pat_rng_num;
			for( ind=0; ind<pat_number; ind++ )
				if ( (ind+1) % interval == 0 ) pat_type[ind] = PAT_TYPE_RNG;
				else pat_type[ind] = PAT_TYPE_SAMP;
		}
		else {
			int interval = pat_number / pat_samp_num;
			for( ind=0; ind<pat_number; ind++ )
				if ( (ind+1) % interval == 0 ) pat_type[ind] = PAT_TYPE_SAMP;
				else pat_type[ind] = PAT_TYPE_RNG;
		}
	}

	/* 2.2.4 Generate the patterns */
	for( ind=0; ind<pat_number; ind++ ) {
		if ( pat_type[ind] == PAT_TYPE_RNG ) {
			pattern[ind] = pattern_allocate( &random, pat_rng_length_mean, pat_rng_length_dev, seq_length, &pat_length[ind] );
			generate_rng_sequence( &random, prob_G, prob_C, prob_A, pattern[ind], pat_length[ind] );
		}
		else if ( pat_type[ind] == PAT_TYPE_SAMP ) {
			pattern[ind] = pattern_allocate( &random, pat_samp_length_mean, pat_samp_length_dev, seq_length, &pat_length[ind] );
#define REGENERATE_SAMPLE_PATTERNS
#ifdef REGENERATE_SAMPLE_PATTERNS
			rng_t random_seq_orig = rng_new( seed );
			generate_sample_sequence( &random, random_seq_orig, prob_G, prob_C, prob_A, seq_length, pat_samp_loc_mean, pat_samp_loc_dev, pattern[ind], pat_length[ind] );
#else
			copy_sample_sequence( &random, sequence, seq_length, pat_samp_loc_mean, pat_samp_loc_dev, pattern[ind], pat_length[ind] );
#endif
		}
		else {
			fprintf(stderr,"\n-- Error internal: Paranoic check! A pattern without type at position %d\n", ind );
			exit( EXIT_FAILURE );
		}
	}
	free( pat_type );

	/* Allocate and move the patterns to the GPU */
	unsigned long *d_pat_length;
	char **d_pattern;
	CUDA_CHECK_FUNCTION( cudaMalloc( &d_pat_length, sizeof(unsigned long) * pat_number ) );
	CUDA_CHECK_FUNCTION( cudaMalloc( &d_pattern, sizeof(char *) * pat_number ) );

	char **d_pattern_in_host = (char **)malloc( sizeof(char*) * pat_number );
	if ( d_pattern_in_host == NULL ) {
		fprintf(stderr,"\n-- Error allocating the patterns structures replicated in the host for size: %d\n", pat_number );
		exit( EXIT_FAILURE );
	}
	for( ind=0; ind<pat_number; ind++ ) {
		CUDA_CHECK_FUNCTION( cudaMalloc( &(d_pattern_in_host[ind]), sizeof(char) * pat_length[ind] ) );
		CUDA_CHECK_FUNCTION( cudaMemcpy( d_pattern_in_host[ind], pattern[ind], pat_length[ind] * sizeof(char), cudaMemcpyHostToDevice ) );
	}
	CUDA_CHECK_FUNCTION( cudaMemcpy( d_pattern, d_pattern_in_host, pat_number * sizeof(char *), cudaMemcpyHostToDevice ) );
	CUDA_CHECK_FUNCTION( cudaMemcpy(d_pat_length, pat_length, sizeof(unsigned long) * pat_number, cudaMemcpyHostToDevice) );


	/* Avoid the usage of arguments to take strategic decisions
	 * In a real case the user only has the patterns and sequence data to analize
	 */
	argc = 0;
	argv = NULL;
	pat_rng_num = 0;
	pat_rng_length_mean = 0;
	pat_rng_length_dev = 0;
	pat_samp_num = 0;
	pat_samp_length_mean = 0;
	pat_samp_length_dev = 0;
	pat_samp_loc_mean = 0;
	pat_samp_loc_dev = 0;
	pat_samp_mix = '0';

	/* 2.3. Other result data and structures */
	int pat_matches = 0;

	/* 2.3.1. Other results related to patterns */
	unsigned long long *pat_found;
	pat_found = (unsigned long long*)malloc( sizeof(unsigned long long) * pat_number );
	if ( pat_found == NULL ) {
		fprintf(stderr,"\n-- Error allocating aux pattern structure for size: %d\n", pat_number );
		exit( EXIT_FAILURE );
	}

	/* 3. Start global timer */
		CUDA_CHECK_FUNCTION( cudaDeviceSynchronize() );
	double ttotal = cp_Wtime();

/*
 *
 * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
 * DO NOT USE OpenMP IN YOUR CODE
 *
 */

	/* 4. Allocazione memoria per la sequenza su host*/
	char *sequence = (char *)malloc( sizeof(char) * seq_length );
	if ( sequence == NULL ) {
		fprintf(stderr,"\n-- Error allocating the sequence for size: %lu\n", seq_length );
		exit( EXIT_FAILURE );
	}

	/* 4.1 Generazione randomica dei caratteri della sequenza */
	random = rng_new( seed );
	generate_rng_sequence( &random, prob_G, prob_C, prob_A, sequence, seq_length);

	/* 4.2 Determinazione dimensione chunk e altri valori utili */
	unsigned long chunkSize = (unsigned long) get_chunk_size(seq_length);
	chunkSize = (chunkSize > seq_length) ? seq_length : chunkSize;

	unsigned long maxPatLength = 0;
	for (int i = 0; i < pat_number; i++) {
		if (pat_length[i] > maxPatLength) maxPatLength = pat_length[i];
	}
	// overlap = maxPatLength - 1 (se >0)
	unsigned long overlap = (maxPatLength > 0) ? (maxPatLength - 1) : 0;
	// Numero di pattern per batch
	unsigned long batchSize = 512;

	/* 4.3 Allocazione memoria e copia dati su GPU */
	char* d_sequence;
	int* d_pat_matches;
	int* d_seq_matches;
	unsigned long long* d_pat_found;

	CUDA_CHECK_FUNCTION(cudaMalloc((void**)&d_sequence, seq_length * sizeof(char)));
	CUDA_CHECK_FUNCTION(cudaMalloc((void**)&d_pat_matches, sizeof(int)));
	CUDA_CHECK_FUNCTION(cudaMalloc((void**)&d_seq_matches, seq_length * sizeof(int)));
	CUDA_CHECK_FUNCTION(cudaMalloc((void**)&d_pat_found, pat_number * sizeof(unsigned long long)));

	CUDA_CHECK_FUNCTION(cudaMemcpy(d_sequence, sequence, seq_length * sizeof(char), cudaMemcpyHostToDevice));
	CUDA_CHECK_FUNCTION(cudaMemset(d_pat_matches, 0, sizeof(int)));

#ifdef DEBUG
	/* DEBUG: Print sequence and patterns */
	printf("-----------------\n");
	printf("Sequence: ");
	for( lind=0; lind<seq_length; lind++ )
		printf( "%c", sequence[lind] );
	printf("\n-----------------\n");
	printf("Patterns: %d ( rng: %d, samples: %d )\n", pat_number, pat_rng_num, pat_samp_num );
	int debug_pat;
	for( debug_pat=0; debug_pat<pat_number; debug_pat++ ) {
		printf( "Pat[%d]: ", debug_pat );
		for( lind=0; lind<pat_length[debug_pat]; lind++ )
			printf( "%c", pattern[debug_pat][lind] );
		printf("\n");
	}
	printf("-----------------\n\n");
#endif // DEBUG

	/* 4.4 Allocazione array seq_matches su host */
	int* seq_matches;
	seq_matches = (int *)malloc( sizeof(int) * seq_length );
	if ( seq_matches == NULL ) {
		fprintf(stderr,"\n-- Error allocating aux sequence structures for size: %lu\n", seq_length );
		exit( EXIT_FAILURE );
	}

	/* 4. Initialize ancillary structures */
	// Inizializza pat_found a NOT_FOUND, ma su device con ULLONG_MAX
	for( ind=0; ind<pat_number; ind++) {
		pat_found[ind] = (unsigned long long)NOT_FOUND;
	}
	// Inizializza seq_matches a NOT_FOUND, ma su device con 0
	for( lind=0; lind<seq_length; lind++) {
		seq_matches[lind] = NOT_FOUND;
	}
	CUDA_CHECK_FUNCTION(cudaMemset(d_seq_matches, 0, seq_length * sizeof(int)));
	for (int i = 0; i < pat_number; i++) {
		pat_found[i] = ULLONG_MAX;
	}
	CUDA_CHECK_FUNCTION(cudaMemcpy(d_pat_found, pat_found, pat_number * sizeof(unsigned long long), cudaMemcpyHostToDevice));


	// Definizione della dimensione del blocco
	unsigned long dimBlockX = 32;
	unsigned long dimBlockY = 12;
	dim3 block(dimBlockX, dimBlockY);

	// Iterazione sui chunk di sequenza
	for (unsigned long seqStart = 0; seqStart < seq_length; seqStart += (chunkSize - overlap)) {
		unsigned long seqEnd = seqStart + chunkSize + overlap;
		seqEnd = (seqEnd > seq_length) ? seq_length : seqEnd;

		unsigned long chunkLen = seqEnd - seqStart;

		// Iterazione sui pattern in batch
		for (int patStart = 0; patStart < pat_number; patStart += batchSize) {
			// Ogni iterazione lavora su un batch di pattern
			int patEnd = patStart + batchSize;
			if (patEnd > pat_number) patEnd = pat_number;
			int numPatternsInThisBatch = patEnd - patStart;

			// Dimensioni griglia (gridX = posizioni sequenza, gridY = pattern)
			unsigned long gridX = (chunkLen + dimBlockX - 1) / dimBlockX;
			unsigned long gridY = (numPatternsInThisBatch + dimBlockY - 1) / dimBlockY;
			dim3 grid(gridX, gridY);

			// Lancio del kernel
			matchPatternsKernel<<<grid, block>>>(
				d_sequence,
				seq_length,
				seqStart,
				chunkLen,
				d_pattern,
				d_pat_length,
				patStart,
				patEnd,
				d_pat_found,
				d_pat_matches
			);
			CUDA_CHECK_KERNEL();
		}		
	}
	CUDA_CHECK_FUNCTION(cudaDeviceSynchronize());
	// Copia di pat_matches da GPU a host
	CUDA_CHECK_FUNCTION(cudaMemcpy(&pat_matches, d_pat_matches, sizeof(int), cudaMemcpyDeviceToHost));
	dim3 blockIncrement(16, 16);
	/*
	* Dimensioni griglia 2D, in cui:
	*   - l'asse X copre i 'pat_number' pattern
	*   - l'asse Y copre la 'maxPatLength' (lunghezza massima pattern)
	*/
	dim3 grid(
		(pat_number + blockIncrement.x - 1) / blockIncrement.x,  // quante "righe" di blocchi per coprire tutti i pattern
		(maxPatLength + blockIncrement.y - 1) / blockIncrement.y   // quante "colonne" di blocchi per coprire la lunghezza massima
	);
	
	/* Questo kernel si occuperà di "incrementare" in parallelo i 
 	 * match sulla sequenza (d_seq_matches) per ogni 
 	 * (pattern, offset nel pattern)
	 */
	incrementMatchesKernel<<<grid, blockIncrement>>>(
		d_pat_found,
		d_pat_length,
		d_seq_matches,
		pat_number,
		seq_length
	);
	CUDA_CHECK_KERNEL();

	CUDA_CHECK_FUNCTION(cudaDeviceSynchronize());
	// Copia di pat_found e seq_matches
	CUDA_CHECK_FUNCTION(cudaMemcpy(pat_found, d_pat_found, pat_number * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
	// Sostituiamo ULLONG_MAX con NOT_FOUND
	for (int i = 0; i < pat_number; i++) {
		if (pat_found[i] == ULLONG_MAX) {
			pat_found[i] = (unsigned long long)NOT_FOUND;
		}
	}
	CUDA_CHECK_FUNCTION(cudaMemcpy(seq_matches, d_seq_matches, seq_length * sizeof(int), cudaMemcpyDeviceToHost));
	// Se > 0 (quindi se ha match), riduco di 1 (come in CPU)
	for (unsigned long i = 0; i < seq_length; i++) {
		if (seq_matches[i] > 0) {
			seq_matches[i]--;
		}else{
			seq_matches[i]=NOT_FOUND;
		}
	}	

	/* 7. Check sums */
	unsigned long long checksum_matches = 0;
	unsigned long checksum_found = 0;
	for( ind=0; ind < pat_number; ind++) {
		if ( pat_found[ind] != (unsigned long long)NOT_FOUND ){
			checksum_found = ( checksum_found + pat_found[ind] ) % CHECKSUM_MAX;
		}
	}
	for( lind=0; lind < seq_length; lind++) {
		if ( seq_matches[lind] != NOT_FOUND )
			checksum_matches = ( checksum_matches + seq_matches[lind] ) % CHECKSUM_MAX;
	}

#ifdef DEBUG
	/* DEBUG: Write results */
	printf("-----------------\n");
	printf("Found start:");
	for( debug_pat=0; debug_pat<pat_number; debug_pat++ ) {
		printf( " %lu", pat_found[debug_pat] );
	}
	printf("\n");
	printf("-----------------\n");
	printf("Matches:");
	for( lind=0; lind<seq_length; lind++ ) 
		printf( " %d", seq_matches[lind] );
	printf("\n");
	printf("-----------------\n");
#endif // DEBUG

	/* Free local resources */	
	free( sequence );
	free( seq_matches );

// TODO FINE 
/*
 *
 * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
 *
 */

	/* 8. Stop global timer */
		CUDA_CHECK_FUNCTION( cudaDeviceSynchronize() );
	ttotal = cp_Wtime() - ttotal;

	/* 9. Output for leaderboard */
	printf("\n");
	/* 9.1. Total computation time */
	printf("Time: %lf\n", ttotal );

	/* 9.2. Results: Statistics */
	printf("Result: %d, %lu, %llu\n\n", 
			pat_matches,
			checksum_found,
			checksum_matches );
		
	/* 10. Free resources */	
	int i;
	for( i=0; i<pat_number; i++ ) free( pattern[i] );
	free( pattern );
	free( pat_length );
	free( pat_found );
	cudaFree(d_pattern);
	cudaFree(d_pat_found);
	cudaFree(d_pat_matches);
	cudaFree(d_sequence);
	cudaFree(d_pattern_in_host);
	cudaFree(d_seq_matches);
	cudaFree(d_pat_length);

	/* 11. End */
	return 0;
}
