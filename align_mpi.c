/*
 * Exact genetic sequence alignment
 * (Using brute force)
 *
 * MPI version
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
 #include<mpi.h>
 
 
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
  *
  */
 
 /*
  * Incrementa i match nella sequenza locale per il pattern trovato, 
  * aggiungendo 1 per ogni nucleotide del pattern alla posizione di match.
  */
  
 void increment_matches(int pat, unsigned long *local_pat_found, unsigned long *pat_length, int *local_seq_matches) {
	 for (unsigned long ind = 0; ind < pat_length[pat]; ind++){
		 local_seq_matches[local_pat_found[pat] + ind]++;
	 }
 }
 
 /*
  * Genera una sequenza casuale di G, C, A, T in base alle probabilità specificate.
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
 * Copia una parte della sequenza esistente per creare un pattern, 
 * scegliendo la posizione di inizio con una distribuzione normale.
 */
 void copy_sample_sequence( rng_t *random, char *sequence, unsigned long seq_length, unsigned long pat_samp_loc_mean, unsigned long pat_samp_loc_dev, char *pattern, unsigned long length) {
	 /* posizione iniziale entro i limiti */
	 unsigned long  location = (unsigned long)rng_next_normal( random, (double)pat_samp_loc_mean, (double)pat_samp_loc_dev );
	 if ( location > seq_length - length ) location = seq_length - length;
	 if ( location <= 0 ) location = 0;
 
	 /* copia i caratteri a partire da 'location' */
	 unsigned long ind; 
	 for( ind=0; ind<length; ind++ )
		 pattern[ind] = sequence[ind+location];
 }
 
 
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
		 MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
	 }
 
	 /* Return results */
	 *new_length = length;
	 return pattern;
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
	 /* 1.0. Init MPI before processing arguments */
	 MPI_Init( &argc, &argv );
	 int rank;
	 MPI_Comm_rank( MPI_COMM_WORLD, &rank );
 
	 /* 1.1. Check minimum number of arguments */
	 if (argc < 15) {
		 fprintf(stderr, "\n-- Error: Not enough arguments when reading configuration from the command line\n\n");
		 show_usage( argv[0] );
		 MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
	 }
 
	 /* 1.2. Read argument values */
	 unsigned long seq_length = atol( argv[1] );
	 float prob_G = atof( argv[2] );
	 float prob_C = atof( argv[3] );
	 float prob_A = atof( argv[4] );
	 if ( prob_G + prob_C + prob_A > 1 ) {
		 fprintf(stderr, "\n-- Error: The sum of G,C,A,T nucleotid probabilities cannot be higher than 1\n\n");
		 show_usage( argv[0] );
		 MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
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
		 MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
	 }
 
	 unsigned long seed = atol( argv[14] );
 
 #ifdef DEBUG
	 /* DEBUG: Print arguments */
	 if ( rank == 0 ) {
		 printf("\nArguments: seq_length=%lu\n", seq_length );
		 printf("Arguments: Accumulated probabilitiy G=%f, C=%f, A=%f, T=1\n", prob_G, prob_C, prob_A );
		 printf("Arguments: Random patterns number=%d, length_mean=%lu, length_dev=%lu\n", pat_rng_num, pat_rng_length_mean, pat_rng_length_dev );
		 printf("Arguments: Sample patterns number=%d, length_mean=%lu, length_dev=%lu, loc_mean=%lu, loc_dev=%lu\n", pat_samp_num, pat_samp_length_mean, pat_samp_length_dev, pat_samp_loc_mean, pat_samp_loc_dev );
		 printf("Arguments: Type of mix: %c, Random seed: %lu\n", pat_samp_mix, seed );
		 printf("\n");
	 }
 #endif // DEBUG
 
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
		 MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
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
		 MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
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
			 MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
		 }
	 }
	 free( pat_type );
 
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
	 //int pat_matches = 0;

	 
	 /* 3. Start global timer */
	 MPI_Barrier( MPI_COMM_WORLD );
	 double ttotal = cp_Wtime();
 
 /*
  *
  * START HERE: DO NOT CHANGE THE CODE ABOVE THIS POINT
  *
  */
	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	/* 2.1. Allocazione e generazione della sequenza (solo il rank 0 la calcola e poi broadcast)*/
	char *sequence = (char *)malloc( sizeof(char) * seq_length );
	if ( sequence == NULL ) {
		fprintf(stderr,"\n-- Error allocating the sequence for size: %lu\n", seq_length );
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
	}
	random = rng_new( seed );
	generate_rng_sequence( &random, prob_G, prob_C, prob_A, sequence, seq_length);
	
	// Condivisione broadcast della sequenza
	MPI_Bcast(sequence, seq_length, MPI_CHAR, 0, MPI_COMM_WORLD);

	// Numero di pattern assegnati a ogni processo
	int patterns_per_proc = pat_number / size;
	int remainder = pat_number % size; // Pattern rimanenti da distribuire
	// Calcola l'intervallo di pattern gestito da ciascun processo
	int local_start = rank * patterns_per_proc + (rank < remainder ? rank : remainder);
	int local_end = local_start + patterns_per_proc + (rank < remainder ? 1 : 0);
	// Numero totale di pattern assegnati al processo
	int local_pat_number = local_end - local_start;

	unsigned long *local_pat_length = (unsigned long *)malloc(local_pat_number * sizeof(unsigned long));
	char **local_pattern = (char **)malloc(local_pat_number * sizeof(char *));
	unsigned long *local_pat_found = (unsigned long *)malloc(local_pat_number * sizeof(unsigned long));

	// Inizializzazione local_pat_found a NOT_FOUND
	for (int i = 0; i < local_pat_number; i++) {
		local_pat_found[i] = NOT_FOUND;
	}

	if (rank == 0) {
		// Rank 0 distribuisce i pattern corretti a ogni processo
		int total_requests = 0;
		for (int dest = 1; dest < size; dest++) {
			int start = dest * patterns_per_proc + (dest < remainder ? dest : remainder);
			int end   = start + patterns_per_proc + (dest < remainder ? 1 : 0);
			// Ogni pattern invia: 1) lunghezza, 2) contenuto -> 2 msg a pattern
			total_requests += 2 * (end - start);
		}
		// Allocazione array di richieste non bloccanti
		MPI_Request *send_requests = (MPI_Request *)malloc(total_requests * sizeof(MPI_Request));
		if (send_requests == NULL) {
			fprintf(stderr, "Errore nell'allocazione di send_requests\n");
			MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		}
		
		int req_index = 0;
		for (int dest = 1; dest < size; dest++) {
			int start = dest * patterns_per_proc + (dest < remainder ? dest : remainder);
			int end   = start + patterns_per_proc + (dest < remainder ? 1 : 0);
			
			for (int i = start; i < end; i++) {
				if (pat_length[i] > INT_MAX) {
					fprintf(stderr, "Pattern length %lu exceeds INT_MAX\n", pat_length[i]);
					MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
				}
				// Invio della lunghezza del pattern (tag 1)
				MPI_Isend(&pat_length[i], 1, MPI_UNSIGNED_LONG, dest, 1, MPI_COMM_WORLD, &send_requests[req_index++]);
				
				// Invio del contenuto del pattern (tag 2)
				MPI_Isend(pattern[i], pat_length[i], MPI_CHAR, dest, 2, MPI_COMM_WORLD, &send_requests[req_index++]);
			}
		}
		
		// Attende che tutte le richieste siano completate
		MPI_Waitall(req_index, send_requests, MPI_STATUSES_IGNORE);
		free(send_requests);

		// Rank 0 copia i pattern che deve gestire localmente
		// I pattern di cui il rank 0 è responsabile sono quelli
		// negli indici [local_start, local_end] dei vettori globali.
		for (int i = local_start; i < local_end; i++) {
			int local_index = i - local_start;  // mappa l'indice globale in quello locale
			local_pat_length[local_index] = pat_length[i];
			local_pattern[local_index] = pattern[i];
		}
	}
	else { // Ricezione dei pattern assegnati

		// Allocazione dell'array di richieste non bloccanti per la ricezione
		MPI_Request *recv_requests = (MPI_Request *)malloc(2 * local_pat_number * sizeof(MPI_Request));
		if (recv_requests == NULL) {
			fprintf(stderr, "Errore nell'allocazione di recv_requests\n");
			MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
		}

		// Ricezione delle lunghezze dei pattern (tag 1)
		for (int i = 0; i < local_pat_number; i++) {
			MPI_Irecv(&local_pat_length[i], 1, MPI_UNSIGNED_LONG, 0, 1, MPI_COMM_WORLD, &recv_requests[i]);
		}
		
		MPI_Waitall(local_pat_number, recv_requests, MPI_STATUSES_IGNORE);
		
		// Allocazione dei buffer per i pattern ricevuti
		for (int i = 0; i < local_pat_number; i++) {
			local_pattern[i] = (char *)malloc(local_pat_length[i] * sizeof(char));
			if (local_pattern[i] == NULL) {
				fprintf(stderr, "Errore nell'allocazione di local_pattern[%d]\n", i);
				MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
			}
		}
		
		// Ricezione dei pattern veri e propri (tag 2)
		for (int i = 0; i < local_pat_number; i++) {
			MPI_Irecv(local_pattern[i], local_pat_length[i], MPI_CHAR, 0, 2, MPI_COMM_WORLD, &recv_requests[local_pat_number + i]);
		}
		// Attende il completamento di tutte le richieste di ricezione
		MPI_Waitall(local_pat_number, &recv_requests[local_pat_number], MPI_STATUSES_IGNORE);
		free(recv_requests);
	}


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
 
	 /* 2.3.2. Inizializza vettore per tracciare i match sulla sequenza */
	int *seq_matches;
	seq_matches = (int *)calloc(seq_length, sizeof(int));
	if ( seq_matches == NULL ) {
		fprintf(stderr,"\n-- Error allocating aux sequence structures for size: %lu\n", seq_length );
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
	}
	// Buffer globale per sommare i match di tutti i processi
	int *global_seq_matches;
	global_seq_matches = (int *)calloc(seq_length, sizeof(int));
	if ( global_seq_matches == NULL ) {
		fprintf(stderr,"\n-- Error allocating aux sequence structures for size: %lu\n", seq_length );
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
	}

	/* 4. Initialize ancillary structures */

	unsigned long *global_pat_found;
	global_pat_found = (unsigned long *)malloc( sizeof(unsigned long) * pat_number );
	if ( global_pat_found == NULL ) {
		fprintf(stderr,"\n-- Error allocating aux pattern structure for size: %d\n", pat_number );
		MPI_Abort( MPI_COMM_WORLD, EXIT_FAILURE );
	}

	for( ind=0; ind<pat_number; ind++) {
		global_pat_found[ind] = (unsigned long)NOT_FOUND;
	}

	for( lind=0; lind<seq_length; lind++) {
		if (rank == 0){
			seq_matches[lind] = (int)NOT_FOUND;
		} else {
			seq_matches[lind] = 0;
		}

		global_seq_matches[lind] = (int)NOT_FOUND;
	}

	/* 5. Search for each pattern */
	unsigned long start;
	int local_matches = 0;
	int global_matches = 0;

	for (int local_pat = 0; local_pat < local_pat_number; local_pat++) {
	
		/* 5.1. For each possible starting position */
		for (start = 0; start <= seq_length - local_pat_length[local_pat]; start++) {
	
			/* 5.1.1. For each pattern element */
			for (lind = 0; lind < local_pat_length[local_pat]; lind++) {
				/* Stop this test when different nucleotides are found */
				if (sequence[start + lind] != local_pattern[local_pat][lind]) break;
			}
	
			/* 5.1.2. Check if the loop ended with a match */
			if (lind == local_pat_length[local_pat]) {
				local_matches++;
				local_pat_found[local_pat] = start;
				break;  // Interrompe la ricerca per questo pattern
			}
		}
	
		/* 5.2. Pattern found */
		if (local_pat_found[local_pat] != (unsigned long)NOT_FOUND) {
			/* Increment the number of pattern matches on the sequence positions */
			increment_matches(local_pat, local_pat_found, local_pat_length, seq_matches);
		}
	}
	
	// Calcola quanti pattern trovati deve inviare ogni processo
	int local_count = 0;
	for (int i = 0; i < local_pat_number; i++) {
		if (local_pat_found[i] != NOT_FOUND) {
			local_count++;
		}
	}

	// Il processo root raccoglie il numero di elementi inviati da ogni processo
	int *recv_counts = NULL;
	int *displs = NULL;

	if (rank == 0) {
		recv_counts = (int *)malloc(size * sizeof(int));
		displs = (int *)malloc(size * sizeof(int));
	}

	// Ogni processo comunica il numero di pattern trovati
	MPI_Gather(&local_count, 1, MPI_INT, recv_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

	if (rank == 0) {
		// Calcola gli offset per ogni processo nel buffer globale
		displs[0] = 0;
		for (int i = 1; i < size; i++) {
			displs[i] = displs[i - 1] + recv_counts[i - 1];
		}
	}

	// Crea un array temporaneo con solo i risultati validi
	unsigned long *local_filtered = (unsigned long *)malloc(local_count * sizeof(unsigned long));
	int index = 0;
	for (int i = 0; i < local_pat_number; i++) {
		if (local_pat_found[i] != NOT_FOUND) {
			local_filtered[index++] = local_pat_found[i];
		}
	}

	// Usa MPI_Gatherv per raccogliere i risultati variabili direttamente in global_pat_found
	MPI_Gatherv(local_filtered, local_count, MPI_UNSIGNED_LONG,
				global_pat_found, recv_counts, displs, MPI_UNSIGNED_LONG,
				0, MPI_COMM_WORLD);

	free(local_filtered);
	if (rank == 0) {
		free(recv_counts);
		free(displs);
	}

	// Riduce il numero totale di pattern trovati (local_matches -> global_matches)
	MPI_Reduce(&local_matches, &global_matches, 1, MPI_UNSIGNED_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	// Somma i match su ogni posizione della sequenza
	MPI_Reduce(seq_matches, global_seq_matches, seq_length, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	/* 7. Check sums (solo il rank 0 li userà) */
	unsigned long checksum_matches = 0;
	unsigned long checksum_found = 0;
	for( ind=0; ind < pat_number; ind++) {
		if ( global_pat_found[ind] != (unsigned long)NOT_FOUND )
			checksum_found = ( checksum_found + global_pat_found[ind] ); 
	}
	checksum_found = checksum_found % CHECKSUM_MAX;

	for( lind=0; lind < seq_length; lind++) {
		if ( global_seq_matches[lind] != NOT_FOUND ){
			checksum_matches = ( checksum_matches + global_seq_matches[lind] );
		}
	}
	checksum_matches = checksum_matches % CHECKSUM_MAX;
 
 #ifdef DEBUG
	 /* DEBUG: Write results */
	 printf("-----------------\n");
	 printf("Found start:");
	 for( debug_pat=0; debug_pat<pat_number; debug_pat++ ) {
		 printf( " %lu", local_pat_found[debug_pat] );
	 }
	 printf("\n");
	 printf("-----------------\n");
	 printf("Matches:");
	 for( lind=0; lind<seq_length; lind++ ) 
		 printf( " %d", global_seq_matches[lind] );
	 printf("\n");
	 printf("-----------------\n");
 #endif // DEBUG
 
	 /* Free local resources */	
	 free( global_seq_matches );
 
 /*
  *
  * STOP HERE: DO NOT CHANGE THE CODE BELOW THIS POINT
  *
  */
 
	 /* 8. Stop global time */
	 MPI_Barrier( MPI_COMM_WORLD );
	 ttotal = cp_Wtime() - ttotal;
 
	 /* 9. Output for leaderboard */
	 if ( rank == 0 ) {
		 printf("\n");
		 /* 9.1. Total computation time */
		 printf("Time: %lf\n", ttotal );
 
		 /* 9.2. Results: Statistics */
		 printf("Result: %d, %lu, %lu\n\n", 
				 global_matches,
				 checksum_found,
				 checksum_matches );
	 }

	 /* 10. Free resources */	
	 int i;
	 for( i=0; i<pat_number; i++ ) free( pattern[i] );
	 free( pattern );
	 free( pat_length );
	 free( local_pat_found );
	 if (rank == 0) {
		 free(sequence);
	 }
 
	 /* 11. End */
	 MPI_Finalize();
	 return 0;
 }
