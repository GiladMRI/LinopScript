/* Copyright 2013, 2016. The Regents of the University of California.
 * Copyright 2015. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors: 
 * 2012, 2015 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2016 Jon Tamir <jtamir@eecs.berkeley.edu>
 */

#include <stdbool.h>
#include <complex.h>

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"
#include "misc/debug.h"

#include "linops/linop.h"
#include "linops/fmac.h"
#include "linops/sampling.h"
#include "linops/someops.h"
#include "num/fft.h"

#ifndef DIMS
#define DIMS 16
#endif

#define MAX_FILES 10
#define ADD_OP const struct linop_s* tmp = Sop;Sop = linop_chain(tmp,NewOp);linop_free(tmp);linop_free(NewOp);

complex float* dataFiles[MAX_FILES];
long Fdims[MAX_FILES][DIMS];

long * getFdims(long i) { return Fdims[i]; }
        
void ReadScriptFiles(char* argv[],long n) {
    long i;
    printf("Reading files\n");
    for(i=0;i<n;i++) {        
        dataFiles[i] = load_cfl(argv[i], DIMS, Fdims[i]);
        printf("Reading %s: %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld\n",argv[i],Fdims[i][0],Fdims[i][1],Fdims[i][2],Fdims[i][3],Fdims[i][4],Fdims[i][5],Fdims[i][6],Fdims[i][7],Fdims[i][8],Fdims[i][9],Fdims[i][10],Fdims[i][11],Fdims[i][12],Fdims[i][13],Fdims[i][14],Fdims[i][15]);
    }
    printf("Finished reading files\n");
}

void ClearReadScriptFiles( char* argv[],long n) {
    long i;
    printf("Clearing files' memory\n");
    for(i=0;i<n;i++) {
        printf("Clearing %s\n",argv[i]);
        unmap_cfl(DIMS, Fdims[i], dataFiles[i]);
    }
    printf("Finished Clearing files' memory\n");
}

const struct linop_s* getLinopScriptFromFile(const char *FN, long CurDims[]) {
    const struct linop_s* Sop = linop_identity_create(DIMS, CurDims);
    
    
    FILE * fp;
	char * line = NULL;
    char *token;
	size_t len = 0;
	ssize_t read;
// 	fp = fopen(argv[1], "r");
    fp = fopen(FN, "r");
	if (fp == NULL) {
		exit(EXIT_FAILURE); }
	while ((read = getline(&line, &len, fp)) != -1) {
// 		printf("Retrieved line of length %zu:\n", read);
//  		printf("%s", line);
		if(read>0) {
			if(line[0]=='#') {
				printf("%s", line);
                continue;
			}
            
            for(int i = 0; line[i]; i++) { line[i] = tolower(line[i]);  }
			token = strtok(line, " ,.-");
//             printf( "Token: XX%sXX\n", token );
            if(strcmp(token,"fft")==0) {
				token = strtok(NULL, " ,.-");
                long FFTFlags=atoi(token);
                printf("Adding: FFT with flag %ld!!\n",FFTFlags);
                
                const struct linop_s* NewOp = linop_fft_create(DIMS, CurDims, FFTFlags);
                ADD_OP
			}
            if(strcmp(token,"ifft")==0) {
				token = strtok(NULL, " ,.-");
                long FFTFlags=atoi(token);
                printf("Adding: IFFT with flag %ld!!\n",FFTFlags);
                
                const struct linop_s* NewOp = linop_ifft_create(DIMS, CurDims, FFTFlags);
                ADD_OP
			}
            if(strcmp(token,"fftc")==0) {
				token = strtok(NULL, " ,.-");
                long FFTFlags=atoi(token);
                printf("Adding: FFTC with flag %ld!!\n",FFTFlags);
                
                const struct linop_s* NewOp = linop_fftc_create(DIMS, CurDims, FFTFlags);
                ADD_OP
			}
            if(strcmp(token,"ifftc")==0) {
				token = strtok(NULL, " ,.-");
                long FFTFlags=atoi(token);
                printf("Adding: IFFTC with flag %ld!!\n",FFTFlags);
                
                const struct linop_s* NewOp = linop_ifftc_create(DIMS, CurDims, FFTFlags);
                ADD_OP
			}
            if(strcmp(token,"fmac")==0) {
				token = strtok(NULL, " ,.-");
                long FileIdx=atoi(token);
                token = strtok(NULL, " ,.-");
                long SquashFlags=atoi(token);
                
                printf("Adding: FMAC with file #%ld squash flag %ld\n",FileIdx,SquashFlags);
                
//                 debug_print_dims(DP_INFO, DIMS, CurDims);
//                 debug_print_dims(DP_INFO, DIMS, Fdims[FileIdx]);
                
                long MergedDims[DIMS];
                md_merge_dims(DIMS, MergedDims, CurDims, Fdims[FileIdx]);
                long NewDims[DIMS];
                md_select_dims(DIMS, ~SquashFlags, NewDims, MergedDims);
                
                long CurFlags=md_nontriv_dims(DIMS,CurDims);
                long NewFlags=md_nontriv_dims(DIMS,NewDims);
                long TFlags=md_nontriv_dims(DIMS,Fdims[FileIdx]);
                
                // update CurDims:
                md_select_dims(DIMS, ~SquashFlags, CurDims, MergedDims);
                
//                 printf("Flags: %ld %ld %ld\n",CurFlags,NewFlags,TFlags);
                const struct linop_s* NewOp = linop_fmac_create(DIMS, MergedDims, 
                    ~NewFlags, ~CurFlags, ~TFlags, dataFiles[FileIdx]);
                ADD_OP
			}
            if(strcmp(token,"transpose")==0) {
				token = strtok(NULL, " ,.-");
                long dim1=atoi(token);
                token = strtok(NULL, " ,.-");
                long dim2=atoi(token);
                printf("Adding: Transpose with dims %ld,%ld\n",dim1,dim2);
                
                const struct linop_s* NewOp = linop_transpose_create(DIMS, CurDims, dim1,dim2);
                ADD_OP
                // update CurDims:
                long tmpDims[DIMS];
                md_copy_dims(DIMS, tmpDims, CurDims);
                md_transpose_dims(DIMS, dim1, dim2, CurDims, tmpDims);
                printf("CurDims: %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld\n",CurDims[0],CurDims[1],CurDims[2],CurDims[3],CurDims[4],CurDims[5],CurDims[6],CurDims[7],CurDims[8],CurDims[9],CurDims[10],CurDims[11],CurDims[12],CurDims[13],CurDims[14],CurDims[15]);
			}
            if(strcmp(token,"print")==0) {
				token = strtok(NULL, " ,.-");
                long msgId=atol(token);
                printf("Adding: print with messageId %ld\n",msgId);
                
                const struct linop_s* NewOp = linop_print_create(DIMS, CurDims, msgId);
                ADD_OP
			}
            if(strcmp(token,"ident")==0) {
				printf("Adding: identity: do nothing\n");
                
                const struct linop_s* NewOp = linop_identity_create(DIMS, CurDims, msgId);
                ADD_OP
			}
		}
	}
	fclose(fp);
	if (line) {
		free(line); }
    printf("\n");
    return Sop;
}

static const char usage_str[] = "<OpScriptTxt> <output> <input0> [<input1> [<input2> [...]]]";
static const char help_str[] =
		"Apply linop -\n"
        "gtry1 <OpScriptTxt> <output> <input0> [<input1> [<input2> [...]]]\n"
		"-----------------------------------------\n"
		"Apply operator script from OpScriptTxt on the input, and save in output\n"
        "Uses other input files if mentioned"
        "Linops:\n"
        "FFT/IFFT/FFTC/IFFTC <FFT_FLAGS>\n"
        "FMAC <Which_file_no> <SQUASH_FLAGS> : multiplies and then sums\n"
        "Transpose <dim1> <dim2> : transposes the dims\n"
        "Print <messageId> : print messageId on frwrd/adjoint/normal calls\n"
        "ident - do nothing";

int main_linopScript(int argc, char* argv[])
{
	bool Normal = false;
    const char* adj_file = NULL;

	const struct opt_s opts[] = {
		OPT_SET('N', &Normal, "Apply Forward+adjoint"),
        OPT_STRING('a', &adj_file, "file", "input file for adjoint"),
	};	

	num_init();
    
    cmdline(&argc, argv, 0, 100, usage_str, help_str, ARRAY_SIZE(opts), opts);
	int num_args = argc - 1;
    
    printf("main_gtry1\n");
    ReadScriptFiles(&argv[3],num_args-2);
        
    long CurDims[DIMS];
	md_select_dims(DIMS, ~0, CurDims, Fdims[0]);
    
    const struct linop_s* Sop =getLinopScriptFromFile(argv[1],CurDims);
    
    /* Ops:
     * FFT FFT_FLAGS
     * IFFT FFT_FLAGS
     * FMAC SQUASH_FLAGS             : md_merge_dims(N, dims, dims1, dims2); md_select_dims(N, ~squash, dimso, dims);
     * Transpose
     * Sum                          : fmax with singleton. Needs flags to squash
     * Permute
     * matrix
     * GRAD?
     * identity
     * sampling
     * realval
     * wavelet
     * finitediff
     * zfinitediff
     * cdiag
     * rdiag
     * resize
     * conv
     * cdf97
     * fftc
     * ifftc
     * nufft
        struct nufft_conf_s nuconf = nufft_conf_defaults;
        nuconf.toeplitz = true;
        nuconf.lowmem = true;
		OPT_SET('K', &nuconf.pcycle, "randshift for NUFFT"),
        const struct linop_s* fft_op = nufft_create2(DIMS, ksp_dims2, coilim_dims, traj_dims, traj, wgs_dims, weights, basis_dims, basis, conf);
     * nudft
     */
        
    
	complex float* out;
            
    long Adjdims[DIMS];
    complex float* adjFile;
    
    printf("Applying the operator\n");
    if(Normal) {
        out = create_cfl(argv[2], DIMS, Fdims[0]);
        linop_normal(Sop, DIMS, Fdims[0], out,	dataFiles[0]);
            
    } else {
        if(adj_file==NULL) {
            out = create_cfl(argv[2], DIMS, CurDims);
            linop_forward(Sop, DIMS, CurDims, out,	DIMS, Fdims[0], dataFiles[0]);
        }
        else {
            
            adjFile = load_cfl(adj_file, DIMS, Adjdims);
            printf("Read %s: %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld\n",adj_file,Adjdims[0],Adjdims[1],Adjdims[2],Adjdims[3],Adjdims[4],Adjdims[5],Adjdims[6],Adjdims[7],Adjdims[8],Adjdims[9],Adjdims[10],Adjdims[11],Adjdims[12],Adjdims[13],Adjdims[14],Adjdims[15]);

            out= create_cfl(argv[2], DIMS, Fdims[0]);

            linop_adjoint(Sop, DIMS, Fdims[0], out,	DIMS, Adjdims, adjFile);
        }
    }

    printf("Saving output\n");
    unmap_cfl(DIMS, CurDims, out);
    
    ClearReadScriptFiles(&argv[3],num_args-2);
    
    if(adj_file!=NULL) {
        printf("Clearing %s\n",adj_file);
        unmap_cfl(DIMS, Adjdims, adjFile);
    }
    
    
	exit(EXIT_SUCCESS);

	return 0;
}


