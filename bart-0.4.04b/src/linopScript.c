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
#include "num/ops.h"
#include "num/iovec.h"

#include "num/casorati.h"


#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"
#include "misc/debug.h"

#include "linops/linop.h"
#include "linops/linopScript.h"
#include "linops/fmac.h"
#include "linops/sampling.h"
#include "linops/someops.h"
#include "linops/sampling.h"
#include "num/fft.h"
#include "misc/utils.h"

#include "noncart/nufft.h"
#include "noncart/nudft.h"
#include "noncart/precond.h"

#include "lowrank/lrthresh.h"

#include "iter/italgos.h"
#include "iter/iter.h"
#include "iter/prox.h"
#include "iter/admm.h"
#include "iter/vec.h"
#include "iter/niht.h"

#include "iter/iter2.h"

#include "iter/prox.h"
#include "iter/thresh.h"

#include "linops/grad.h"
#include "linops/sum.h"
#include "linops/waveop.h"

#include "wavelet/wavthresh.h"


#include "misc/mri.h"
#include "misc/utils.h"

#include "grecon/optreg.h"

#ifndef DIMS
#define DIMS 16
#endif

#define MAX_LINOPS 20
#define MAX_FILES 10
// #define NEW_OP_POINTER  const struct linop_s* NewOp
#define NEW_OP_POINTER  LinopsVec[LinopCounter]
// linop_free(lop);
#define ADD_OP Sop = linop_chain(Sop,LinopsVec[LinopCounter]);md_copy_dims(DIMS, CurDims, linop_codomain(LinopsVec[LinopCounter])->dims);debug_printf(DP_INFO,"OutDims: ");debug_print_dims(DP_INFO,DIMS,CurDims);LinopCounter++;
// #define ADD_OP const struct linop_s* tmp=Sop;  Sop = linop_chain(tmp,LinopsVec[LinopCounter]);md_copy_dims(DIMS, CurDims, linop_codomain(LinopsVec[LinopCounter])->dims);debug_printf(DP_INFO,"OutDims: ");debug_print_dims(DP_INFO,DIMS,linop_codomain(LinopsVec[LinopCounter])->dims);LinopCounter++;
// #define ADD_OP Sop = linop_chain(Sop,NewOp);md_copy_dims(DIMS, CurDims, linop_codomain(Sop)->dims);LinopsVec[LinopCounter]=NewOp;debug_printf(DP_INFO,"OutDims: ");debug_print_dims(DP_INFO,DIMS,CurDims);LinopCounter++;
// #define ADD_OP const struct linop_s* tmp = Sop;Sop = linop_chain(tmp,NewOp);linop_free(tmp);linop_free(NewOp);md_copy_dims(DIMS, CurDims, linop_codomain(Sop)->dims);

complex float* dataFiles[MAX_FILES];
long Fdims[MAX_FILES][DIMS];

const struct linop_s* LinopsVec[MAX_LINOPS];
long LinopCounter=0;

long * getFdims(long i) { return Fdims[i]; }
complex float* getDataFile(long i) { return dataFiles[i]; }

void FreeLinops() {
    debug_printf(DP_INFO,"Freeing %ld linops\n",LinopCounter);
    for(long i=0;i<LinopCounter;i++) {
        linop_free(LinopsVec[i]);
    }
    debug_printf(DP_INFO,"FreeLinops done\n");
}
void ReadScriptFiles(char* argv[],long n) {
    long i;
    debug_printf(DP_INFO,"Reading files\n");
    for(i=0;i<n;i++) {        
        dataFiles[i] = load_cfl(argv[i], DIMS, Fdims[i]);
        printf("Reading %s: %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld\n",argv[i],Fdims[i][0],Fdims[i][1],Fdims[i][2],Fdims[i][3],Fdims[i][4],Fdims[i][5],Fdims[i][6],Fdims[i][7],Fdims[i][8],Fdims[i][9],Fdims[i][10],Fdims[i][11],Fdims[i][12],Fdims[i][13],Fdims[i][14],Fdims[i][15]);
    }
    debug_printf(DP_INFO,"Finished reading files\n");
}

void ClearReadScriptFiles( char* argv[],long n) {
    long i;
    debug_printf(DP_INFO,"Clearing files' memory\n");
    for(i=0;i<n;i++) {
        debug_printf(DP_INFO,"Clearing %s\n",argv[i]);
        unmap_cfl(DIMS, Fdims[i], dataFiles[i]);
    }
    debug_printf(DP_INFO,"Finished Clearing files' memory\n");
}

/*const struct linop_s* getOpFromFile(const char *FN, long CurDims[],FILE * fp) {
    const struct linop_s* Sop = linop_identity_create(DIMS, CurDims);
}*/

const struct linop_s* getLinopScriptFromFile(const char *FN, long CurDims[]) {
    LinopCounter=0;
    const struct linop_s* Sop = linop_identity_create(DIMS, CurDims);

    const struct linop_s* SopBackup;
    
    FILE * fp;
	char * line = NULL;
    char *token;
	size_t len = 0;
	ssize_t read;
    debug_printf(DP_INFO,"getLinopScriptFromFile start\n");

    long StartDims[DIMS];
    md_copy_dims(DIMS, StartDims, CurDims);
    
// 	fp = fopen(argv[1], "r");
    fp = fopen(FN, "r");
	if (fp == NULL) {
        debug_printf(DP_ERROR, "Couldn't open script file!!!");
        exit(EXIT_FAILURE); }
    // debug_printf(DP_INFO,"--------------\n");
    
    bool NormalOp=false;

	while ((read = getline(&line, &len, fp)) != -1) {
// 		debug_printf(DP_INFO,"Retrieved line of length %zu:\n", read);
 		// debug_printf(DP_INFO,"LINE READ: %s", line);
		if(read>0) {
			if(line[0]=='#') {
				debug_printf(DP_INFO,"%s", line);
                continue;
			}
            
            for(int i = 0; line[i]; i++) { line[i] = tolower(line[i]);  }
			token = strtok(line, " ,.-");
//             printf( "Token: XX%sXX\n", token );
            if(strcmp(token,"fft")==0) {
				token = strtok(NULL, " ,.-");
                long FFTFlags=atoi(token);
                debug_printf(DP_INFO,"Linop %ld: Adding: FFT with flag %ld\n",LinopCounter,FFTFlags);
                
                NEW_OP_POINTER = linop_fft_create(DIMS, CurDims, FFTFlags);
                ADD_OP
			}
            if(strcmp(token,"ifft")==0) {
				token = strtok(NULL, " ,.-");
                long FFTFlags=atoi(token);
                debug_printf(DP_INFO,"Linop %ld: Adding: IFFT with flag %ld\n",LinopCounter,FFTFlags);
                
                NEW_OP_POINTER = linop_ifft_create(DIMS, CurDims, FFTFlags);
                ADD_OP
			}
            if(strcmp(token,"fftc")==0) {
				token = strtok(NULL, " ,.-");
                long FFTFlags=atoi(token);
                debug_printf(DP_INFO,"Linop %ld: Adding: FFTC with flag %ld\n",LinopCounter,FFTFlags);
                
                NEW_OP_POINTER = linop_fftc_create(DIMS, CurDims, FFTFlags);
                ADD_OP
			}
            if(strcmp(token,"ifftc")==0) {
				token = strtok(NULL, " ,.-");
                long FFTFlags=atoi(token);
                debug_printf(DP_INFO,"Linop %ld: Adding: IFFTC with flag %ld\n",LinopCounter,FFTFlags);
                
                NEW_OP_POINTER = linop_ifftc_create(DIMS, CurDims, FFTFlags);
                ADD_OP
			}
            if(strcmp(token,"fmac")==0) {
				token = strtok(NULL, " ,.-");
                long FileIdx=atoi(token);
                token = strtok(NULL, " ,.-");
                long SquashFlags=atoi(token);
                
                debug_printf(DP_INFO,"Linop %ld: Adding: FMAC with file #%ld squash flag %ld\n",LinopCounter,FileIdx,SquashFlags);
                
                debug_print_dims(DP_INFO, DIMS, CurDims);
                debug_print_dims(DP_INFO, DIMS, Fdims[FileIdx]);
                
                long MergedDims[DIMS];
                md_merge_dims(DIMS, MergedDims, CurDims, Fdims[FileIdx]);
                long NewDims[DIMS];
                md_select_dims(DIMS, ~SquashFlags, NewDims, MergedDims);
                
                long CurFlags=md_nontriv_dims(DIMS,CurDims);
                long NewFlags=md_nontriv_dims(DIMS,NewDims);
                long TFlags=md_nontriv_dims(DIMS,Fdims[FileIdx]);
                
                // update CurDims:
                // md_copy_dims(DIMS, CurDims, NewDims);
                // md_select_dims(DIMS, ~SquashFlags, CurDims, MergedDims);
                
                // debug_printf(DP_INFO,"Flags: %ld %ld %ld\n",CurFlags,NewFlags,TFlags);
                NEW_OP_POINTER = linop_fmac_create(DIMS, MergedDims, 
                    ~NewFlags, ~CurFlags, ~TFlags, dataFiles[FileIdx]);
                ADD_OP
			}
            if(strcmp(token,"transpose")==0) {
				token = strtok(NULL, " ,.-");
                long dim1=atoi(token);
                token = strtok(NULL, " ,.-");
                long dim2=atoi(token);
                debug_printf(DP_INFO,"Linop %ld: Adding: Transpose with dims %ld,%ld\n",LinopCounter,dim1,dim2);
                
                NEW_OP_POINTER = linop_transpose_create(DIMS, CurDims, dim1,dim2);
                ADD_OP
                // // update CurDims:
                // long tmpDims[DIMS];
                // md_copy_dims(DIMS, tmpDims, CurDims);
                // md_transpose_dims(DIMS, dim1, dim2, CurDims, tmpDims);
                // debug_printf(DP_INFO,"CurDims: %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld\n",CurDims[0],CurDims[1],CurDims[2],CurDims[3],CurDims[4],CurDims[5],CurDims[6],CurDims[7],CurDims[8],CurDims[9],CurDims[10],CurDims[11],CurDims[12],CurDims[13],CurDims[14],CurDims[15]);
			}
            if(strcmp(token,"print")==0) {
				token = strtok(NULL, " ,.-");
                long msgId=atol(token);
                debug_printf(DP_INFO,"Linop %ld: Adding: print with messageId %ld\n",LinopCounter,msgId);
                
                NEW_OP_POINTER = linop_print_create(DIMS, CurDims, msgId);
                ADD_OP
			}
            if(strcmp(token,"ident")==0) {
				debug_printf(DP_INFO,"Linop %ld: Adding: identity: do nothing\n",LinopCounter);
                
                NEW_OP_POINTER = linop_identity_create(DIMS, CurDims);
                ADD_OP
			}
            if(strcmp(token,"samp")==0) { 
				token = strtok(NULL, " ,.-");
                long FileIdx=atoi(token);
                
                debug_printf(DP_INFO,"Linop %ld: Adding: Sampling with file #%ld\n",LinopCounter,FileIdx);
                NEW_OP_POINTER = linop_samplingGeneral_create(CurDims, Fdims[FileIdx], dataFiles[FileIdx]);
                ADD_OP
			}
            if(strcmp(token,"part")==0) { 
				token = strtok(NULL, " ,.-");
                long dim1=atoi(token);
                token = strtok(NULL, " ,.-");
                long dim2=atoi(token);
                token = strtok(NULL, " ,.-");
                long K=atoi(token);
                debug_printf(DP_INFO,"Linop %ld: Adding: Partition with dims %ld,%ld and K=%ld\n",LinopCounter,dim1,dim2,K);
                
                NEW_OP_POINTER = linop_PartitionDim_create(DIMS, CurDims, dim1, dim2, K);
                ADD_OP
			}
            if(strcmp(token,"hankel")==0) { 
				token = strtok(NULL, " ,.-");
                long dim1=atoi(token);
                token = strtok(NULL, " ,.-");
                long dim2=atoi(token);
                token = strtok(NULL, " ,.-");
                long K=atoi(token);
                debug_printf(DP_INFO,"Linop %ld: Adding: Hankel with dims %ld,%ld and K=%ld\n",LinopCounter,dim1,dim2,K);
                
                NEW_OP_POINTER = linop_Hankel_create(DIMS, CurDims, dim1, dim2, K);
                ADD_OP
			}
            if(strcmp(token,"nufft")==0) {
                token = strtok(NULL, " ,.");
                long TrajFileIdx=atoi(token);
                token = strtok(NULL, " ,.");
                long WeightsFileIdx=atoi(token);
                token = strtok(NULL, " ,.");
                long BasisFileIdx=atoi(token);
                token = strtok(NULL, " ,.");
                long NUFlags=atoi(token);
                token = strtok(NULL, " ,.");
                long Toep=atoi(token);
                token = strtok(NULL, " ,.");
                long pcycle=atoi(token);
                token = strtok(NULL, " ,.");
                long periodic=atoi(token);
                token = strtok(NULL, " ,.");
                long lowmem=atoi(token);

                debug_printf(DP_INFO,"Linop %ld: Adding: NUFFT  with file #%ld Weights file %ld Basis file %ld Flags %ld Toeplitz %ld pcycle %ld periodic %ld lowmem %ld\n",
                    LinopCounter,TrajFileIdx,WeightsFileIdx,BasisFileIdx,NUFlags,Toep,pcycle,periodic,lowmem);
                debug_printf(DP_INFO,"----------- NUFFT Trajectory should be [3, Readout, Spokes] !!!\n");
                
                debug_print_dims(DP_INFO, DIMS, CurDims);
                debug_print_dims(DP_INFO, DIMS, Fdims[TrajFileIdx]);
                
                struct nufft_conf_s nuconf = nufft_conf_defaults;
                nuconf.flags = NUFlags;
                nuconf.toeplitz = Toep>0;
                nuconf.pcycle = pcycle>0;
                nuconf.periodic = periodic>0;
                nuconf.lowmem = lowmem>0;

                long ksp_dims[DIMS];
                md_select_dims(DIMS, PHS1_FLAG|PHS2_FLAG, ksp_dims, Fdims[TrajFileIdx]);
                md_copy_dims(DIMS - 3, ksp_dims + 3, CurDims + 3);

                // NEW_OP_POINTER;
                if(BasisFileIdx<0 && WeightsFileIdx<0) {
                    NEW_OP_POINTER= nufft_create(DIMS, ksp_dims, CurDims, Fdims[TrajFileIdx], dataFiles[TrajFileIdx], NULL, nuconf);
                } else if(BasisFileIdx<0 && WeightsFileIdx>=0) {
                    NEW_OP_POINTER= nufft_create2(DIMS, ksp_dims, CurDims, Fdims[TrajFileIdx], dataFiles[TrajFileIdx], Fdims[WeightsFileIdx],dataFiles[WeightsFileIdx],NULL,NULL, nuconf);
                } else if(BasisFileIdx>=0 && WeightsFileIdx<0) {
                    NEW_OP_POINTER= nufft_create2(DIMS, ksp_dims, CurDims, Fdims[TrajFileIdx], dataFiles[TrajFileIdx], NULL,NULL, Fdims[BasisFileIdx],dataFiles[BasisFileIdx], nuconf);
                } else if(BasisFileIdx>=0 && WeightsFileIdx>=0) {
                    NEW_OP_POINTER= nufft_create2(DIMS, ksp_dims, CurDims, Fdims[TrajFileIdx], dataFiles[TrajFileIdx],
                                        Fdims[WeightsFileIdx],dataFiles[WeightsFileIdx], Fdims[BasisFileIdx],dataFiles[BasisFileIdx], nuconf);
                }
                ADD_OP
            }
            if(strcmp(token,"wavelet")==0) { 
				token = strtok(NULL, " ,.-");
                long xflags=atoi(token);
                token = strtok(NULL, " ,.-");
                long randshift=atoi(token);
                debug_printf(DP_INFO,"Linop %ld: Adding: Wavelet with xflags %ld randshift %ld\n",LinopCounter,xflags,randshift);

                long cur_strs[DIMS];
			    md_calc_strides(DIMS, cur_strs, CurDims, CFL_SIZE);

                long minsize[DIMS] = { [0 ... DIMS - 1] = 1 };
                minsize[0] = MIN(CurDims[0], 16);
                minsize[1] = MIN(CurDims[1], 16);
                minsize[2] = MIN(CurDims[2], 16);


                unsigned int wflags = 0;
                unsigned int wxdim = 0;
                for (unsigned int i = 0; i < DIMS; i++) {

                    if ((1 < CurDims[i]) && MD_IS_SET(xflags, i)) {

                        wflags = MD_SET(wflags, i);
                        minsize[i] = MIN(CurDims[i], 16);
                        wxdim += 1;
                    }
                }
                
                NEW_OP_POINTER = linop_wavelet_create(DIMS, wflags, CurDims, cur_strs, minsize, randshift>0);
                ADD_OP
			}
            if(strcmp(token,"resize")==0) { 
				token = strtok(NULL, " ,.-");
                long FileIdx=atoi(token);
                
                debug_printf(DP_INFO,"Linop %ld: Adding: resize by #%ld\n",LinopCounter,FileIdx);
                NEW_OP_POINTER = linop_resize_create(DIMS, dataFiles[FileIdx],CurDims);
                ADD_OP
			}
            if(strcmp(token,"dblszc")==0) { 
				token = strtok(NULL, " ,.-");
                long dblFlags=atoi(token);

                long dbl_dims[DIMS];
                md_copy_dims(DIMS,dbl_dims,CurDims);
                for (unsigned int i = 0; i < DIMS; i++) {
                    if (MD_IS_SET(dblFlags, i)) {
                        dbl_dims[i] *= 2;
                    }
                }

                debug_printf(DP_INFO,"Linop %ld: Adding: dblszc flags  #%ld\n",LinopCounter,dblFlags);

                debug_printf(DP_INFO,"dbl_dims: ");
                debug_print_dims(DP_INFO,DIMS,dbl_dims);
                
                NEW_OP_POINTER = linop_resize_create(DIMS, dbl_dims,CurDims);
                ADD_OP
			}
            if(strcmp(token,"halfszc")==0) { 
				token = strtok(NULL, " ,.-");
                long halfFlags=atoi(token);

                long half_dims[DIMS];
                md_copy_dims(DIMS,half_dims,CurDims);
                for (unsigned int i = 0; i < DIMS; i++) {
                    if (MD_IS_SET(halfFlags, i)) {
                        half_dims[i] /= 2;
                    }
                }

                debug_printf(DP_INFO,"Linop %ld: Adding: halfszc flags  #%ld\n",LinopCounter,halfFlags);

                debug_printf(DP_INFO,"half_dims: ");
                debug_print_dims(DP_INFO,DIMS,half_dims);
                
                NEW_OP_POINTER = linop_resize_create(DIMS, half_dims,CurDims);
                ADD_OP
			}
            if(strcmp(token,"dblsz")==0) { 
				token = strtok(NULL, " ,.-");
                long dblFlags=atoi(token);

                long dbl_dims[DIMS];
                md_copy_dims(DIMS,dbl_dims,CurDims);
                for (unsigned int i = 0; i < DIMS; i++) {
                    if (MD_IS_SET(dblFlags, i)) {
                        dbl_dims[i] *= 2;
                    }
                }

                debug_printf(DP_INFO,"Linop %ld: Adding: dblsz flags  #%ld\n",LinopCounter,dblFlags);

                debug_printf(DP_INFO,"dbl_dims: ");
                debug_print_dims(DP_INFO,DIMS,dbl_dims);
                
                NEW_OP_POINTER = linop_resizeBase_create(DIMS, dbl_dims,CurDims);
                ADD_OP
			}
            if(strcmp(token,"halfsz")==0) { 
				token = strtok(NULL, " ,.-");
                long halfFlags=atoi(token);

                long half_dims[DIMS];
                md_copy_dims(DIMS,half_dims,CurDims);
                for (unsigned int i = 0; i < DIMS; i++) {
                    if (MD_IS_SET(halfFlags, i)) {
                        half_dims[i] /= 2;
                    }
                }

                debug_printf(DP_INFO,"Linop %ld: Adding: halfsz flags  #%ld\n",LinopCounter,halfFlags);

                debug_printf(DP_INFO,"half_dims: ");
                debug_print_dims(DP_INFO,DIMS,half_dims);
                
                NEW_OP_POINTER = linop_resizeBase_create(DIMS, half_dims,CurDims);
                ADD_OP
			}
            if(strcmp(token,"f")==0) {
                token = strtok(NULL, " ,.-");
                long Idx=atoi(token);
                debug_printf(DP_INFO,"Adding forward of linop #%ld\n",Idx);

                debug_printf(DP_INFO,"CurDims: ");
                debug_print_dims(DP_INFO,DIMS,CurDims);

                debug_printf(DP_INFO,"LinopIdx dims: ");
                debug_print_dims(DP_INFO,DIMS,linop_domain(LinopsVec[Idx]));
                debug_print_dims(DP_INFO,DIMS,linop_codomain(LinopsVec[Idx]));
                
                Sop = linop_chain(Sop,LinopsVec[Idx]);
                md_copy_dims(DIMS, CurDims, linop_codomain(Sop)->dims);
            }
            if(strcmp(token,"a")==0) {
                token = strtok(NULL, " ,.-");
                long Idx=atoi(token);

                debug_printf(DP_INFO,"Adding adjoint of linop #%ld\n",Idx);

                debug_printf(DP_INFO,"CurDims: ");
                debug_print_dims(DP_INFO,DIMS,CurDims);

                debug_printf(DP_INFO,"LinopIdx dims: ");
                debug_print_dims(DP_INFO,DIMS,linop_domain(LinopsVec[Idx])->dims);
                debug_print_dims(DP_INFO,DIMS,linop_codomain(LinopsVec[Idx])->dims);

                debug_printf(DP_INFO,"chain\n");
                Sop = linop_chainAdj(Sop,LinopsVec[Idx]);
                md_copy_dims(DIMS, CurDims, linop_codomain(Sop)->dims);
            }
            if(strcmp(token,"n")==0) {
                token = strtok(NULL, " ,.-");
                long Idx=atoi(token);
                debug_printf(DP_INFO,"Adding Normal of linop #%ld\n",Idx);
                Sop = linop_chainNormal(Sop,LinopsVec[Idx]);
                md_copy_dims(DIMS, CurDims, linop_codomain(Sop)->dims);
            }
            if(strcmp(token,"normal")==0) {
                debug_printf(DP_INFO,"---------\nMoving to normal\n----------------\n");
                NormalOp=true;
                SopBackup=Sop;
                md_copy_dims(DIMS, CurDims, StartDims);
                Sop = linop_identity_create(DIMS, CurDims);
            }
		}
	}
	fclose(fp);
	if (line) {
		free(line); }
    if(NormalOp) {
        const struct linop_s* tmp=Sop;
        Sop=linop_PutFowrardOfBInNormalOfA(SopBackup, tmp);
        linop_free(tmp);
    }
    FreeLinops();
    debug_printf(DP_INFO,"getLinopScriptFromFile end\n");
    return Sop;
}

static const char usage_str[] = "<OpScriptTxt> <StartDims> <input> [<file0> [<file> [<file2> [...]]]] <output>";
static const char help_str[] =
		"Apply linop from script -\n"
        "linopScript <OpScriptTxt> <StartDims> <input> [<file0> [<file> [<file2> [...]]]] <output>\n"
		"-----------------------------------------\n"
		"Apply operator script from OpScriptTxt on the input, and save in output\n"
        "Uses other files if mentioned\n"
        "Linops:\n"
        "FFT/IFFT/FFTC/IFFTC <FFT_FLAGS>\n"
        "FMAC <Which_file_no> <SQUASH_FLAGS> : multiplies and then sums\n"
        "Transpose <dim1> <dim2> : transposes the dims\n"
        "Print <messageId> : print messageId on frwrd/adjoint/normal calls\n"
        "ident - do nothing\n"
        "Samp <Which_file_no> : Sampling is multiplication by binary map - so forward=adjoint=normal\n"
        "Part Dim1 Dim2 K\n"
        "Hankel Dim1 Dim2 K\n"
        "resize FileIdx : File contains new size\n"
        "dblsz/halfsz/dblszc/halfszc Flags : Flags for dims to double/half\n"
        "NUFFT TrajFileIdx WeightsFileIdx BasisFileIdx NUFlags ToepBool pcycleBool periodicBool lowmemBool : TrajFile should be [3 readout spokes]. Bool 0/1. NUFT defaults are false.";

int main_linopScript(int argc, char* argv[])
{
    LinopCounter=0;

	bool Normal = false;
    bool Adj = false;
    unsigned int fftmod_flags = 0;

	const struct opt_s opts[] = {
		OPT_SET('N', &Normal, "Apply normal"),
        OPT_SET('A', &Adj, "Apply adjoint"),
        OPT_UINT('j', &fftmod_flags, "fftmod_flags", "flags for fftmod_flags of (k-space?) input (if forward/normal) and (sensitivity?) file0"),
	};	

	num_init();
    
    cmdline(&argc, argv, 0, 100, usage_str, help_str, ARRAY_SIZE(opts), opts);
	int num_args = argc - 1;
    
    debug_printf(DP_INFO,"main_linopScript\n");
    debug_printf(DP_INFO,"AAA %d\n",num_args);
    debug_printf(DP_INFO,"%s\n",argv[0]); // "linopScript"
    debug_printf(DP_INFO,"%s\n",argv[1]); // script file
    debug_printf(DP_INFO,"%s\n",argv[2]); // input dims
    debug_printf(DP_INFO,"%s\n",argv[3]); // input
    debug_printf(DP_INFO,"Out: %s\n",argv[num_args]); // input
    debug_printf(DP_INFO,"BBB\n");
    
    ReadScriptFiles(&argv[4],num_args-4);
    
    // fftmod(DIMS, getFdims(1), fftmod_flags, getDataFile(1), getDataFile(1));
    
    debug_printf(DP_INFO,"input dims: %s\n",argv[3]);
    debug_printf(DP_INFO,"XXXXXXXXXXXXXX\n",argv[3]);
    long inputDims_dims[DIMS];
    long input_dims[DIMS];
    complex float* inputDims = load_cfl(argv[2], DIMS, inputDims_dims);
    
    debug_printf(DP_INFO,"inputDims_dims: ");
    debug_print_dims(DP_INFO,DIMS,inputDims_dims);
    
    md_copy_dims(DIMS, input_dims, inputDims_dims);
    for(long d=0;d<inputDims_dims[0]*inputDims_dims[1];d++) {
        // debug_printf(DP_INFO,"d %d\n",d);
        input_dims[d]=inputDims[d];
    }
    // md_copy_dims(DIMS, input_dims, inputDims);
    debug_printf(DP_INFO,"input_dims: ");
    debug_print_dims(DP_INFO,DIMS,input_dims);
        
    long inputF_dims[DIMS];
    complex float* input=load_cfl(argv[3], DIMS, inputF_dims);

    long CurDims[DIMS];
	md_copy_dims(DIMS, CurDims, input_dims);
    
    debug_printf(DP_INFO,"Reading script:\n");
    const struct linop_s* Sop =getLinopScriptFromFile(argv[1],CurDims);
    
    /* Ops:
     * FFT FFT_FLAGS
     * IFFT FFT_FLAGS
     * FMAC SQUASH_FLAGS             : md_merge_dims(N, dims, dims1, dims2); md_select_dims(N, ~squash, dimso, dims);
     * identity
     * Transpose                       ? void md_copy2(unsigned int D, const long dim[D], const long ostr[D], void* optr, const long istr[D], const void* iptr, size_t size)
     * sampling
     * nufft
     
     * Sum                          : fmax with singleton. Needs flags to squash
     * Permute
     * matrix
     * GRAD?
     * realval
     * wavelet
     * finitediff
     * zfinitediff
     * cdiag static struct linop_s* linop_gdiag_create(unsigned int N, const long dims[N], unsigned int flags, const complex float* diag, bool rdiag)
     * rdiag
     * resize : pad and crop
     * conv
     * cdf97
     * nudft
     */
        
    
	complex float* out;
    
    if(!Adj && !Normal)  // Forward: Imag->k
    {
        debug_printf(DP_INFO,"Applying the operator\n");
        debug_printf(DP_INFO, "From linop out:");
        debug_print_dims(DP_INFO,linop_codomain(Sop)->N,linop_codomain(Sop)->dims);
        md_copy_dims(DIMS, CurDims, linop_codomain(Sop)->dims);
        out = create_cfl(argv[num_args], DIMS, CurDims);
        linop_forward(Sop, DIMS, CurDims, out,	DIMS, input_dims, input);
    } else if(!Normal) { // adjoint
        debug_printf(DP_INFO,"Applying the operator - adjoint\n");
        out = create_cfl(argv[num_args], DIMS, input_dims);
        linop_adjoint(Sop, DIMS, input_dims, out,	DIMS, inputF_dims, input);
    } else { // Normal
        debug_printf(DP_INFO,"Applying the operator - Normal\n");
        out = create_cfl(argv[num_args], DIMS, input_dims);
        linop_normal(Sop, DIMS, input_dims, out, input);
    }

    // To apply a proximal operator
    /*
    long blkdims[DIMS];
    for(long d=0;d<16;d++) {
        blkdims[d]=1;
    }

    int remove_mean = 0;

    bool randshift=false;
    bool overlapping_blocks=false;
    float lambda=0.01;
    unsigned int xflags=51;
    const struct operator_p_s* prox_op=lrthresh_create(CurDims, randshift, xflags, blkdims, lambda, false, remove_mean, overlapping_blocks);

                    // const struct operator_s* opop=operator_p_upcast(prox_op);

    const struct operator_s* opop=opFromOpps(prox_op,0.3);
    operator_apply(opop, DIMS, CurDims, out, DIMS, CurDims, out);

                    // struct iter_op_p_s a_prox_ops = OPERATOR_P2ITOP(prox_op);

                    // a_prox_ops.fun(a_prox_ops.data, 0.3, out, out);
    */



    // fftmod(DIMS, Adjdims, fftmod_flags, adjFile, adjFile);

    // linop_adjoint(Sop, DIMS, input_dims, outadjFile);
   
//    FreeLinops();

    debug_printf(DP_INFO,"Saving output\n");
    unmap_cfl(DIMS, linop_codomain(Sop)->dims, out);
    
    ClearReadScriptFiles(&argv[4],num_args-4);
    unmap_cfl(DIMS, inputF_dims, input);
    
    // xfree(adj_file);
    
	exit(EXIT_SUCCESS);

	return 0;
}






/*
#include <math.h>long blkdims[MAX_LEV][DIMS];
#include "mex.h"


#define	T_IN	plong blkdims[MAX_LEV][DIMS];
#define	Y_IN	plong blkdims[MAX_LEV][DIMS];



#define	YP_OUT	plhs[0]

#if !defined(MAX)
#define	MAX(A, B)	((A) > (B) ? (A) : (B))
#endif

#if !defined(MIN)
#define	MIN(A, B)	((A) < (B) ? (A) : (B))
#endif

static	double	mu = 1/82.45;
static	double	mus = 1 - 1/82.45;


static void yprime(
		   double	yp[],
		   double	*t,
 		   double	y[]
		   )
{
    double	r1,r2;

    (void) t;     
    r1 = sqrt((y[0]+mu)*(y[0]+mu) + y[2]*y[2]); 
    r2 = sqrt((y[0]-mus)*(y[0]-mus) + y[2]*y[2]);

    if (r1 == 0.0 || r2 == 0.0 ){
        mexWarnMsgIdAndTxt( "MATLAB:yprime:divideByZero", 
                "Division by zero!\n");
    }

    yp[0] = y[1];
    yp[1] = 2*y[3]+y[0]-mus*(y[0]+mu)/(r1*r1*r1)-mu*(y[0]-mus)/(r2*r2*r2);
    yp[2] = y[3];
    yp[3] = -2*y[1] + y[2] - mus*y[2]/(r1*r1*r1) - mu*y[2]/(r2*r2*r2);
    return;
}


void mexFunction( int nlhs, mxArray *plhs[],  int nrhs, const mxArray*prhs[] )
     
{ 
    double *yp; 
    double *t,*y; 
    size_t m,n; 


    if (nrhs != 2) { 
	    mexErrMsgIdAndTxt( "MATLAB:yprime:invalidNumInputs",
                "Two input arguments required."); 
    } else if (nlhs > 1) {
	    mexErrMsgIdAndTxt( "MATLAB:yprime:maxlhs",
                "Too many output arguments."); 
    } 

    if( !mxIsDouble(T_IN) || mxIsComplex(T_IN)) {
      mexErrMsgIdAndTxt( "MATLAB:yprime:invalidT",
              "First input argument must be a real matrix.");
    }

    if( !mxIsDouble(Y_IN) || mxIsComplex(Y_IN)) {
      mexErrMsgIdAndTxt( "MATLAB:yprime:invalidY",
              "Second input argument must be a real matrix.");
    }


    m = mxGetM(Y_IN); 
    n = mxGetN(Y_IN);
    if (!mxIsDouble(Y_IN) || mxIsComplex(Y_IN) || mxIsSparse(Y_IN) || 
	(MAX(m,n) != 4) || (MIN(m,n) != 1)) { 
	    mexErrMsgIdAndTxt( "MATLAB:yprime:invalidY",
                "YPRIME requires that Y be a 4 x 1 vector."); 
    } 

    YP_OUT = mxCreateDoubleMatrix( (mwSize)m, (mwSize)n, mxREAL); 

    yp = mxGetPr(YP_OUT);

    t = mxGetPr(T_IN); 
    y = mxGetPr(Y_IN);

    yprime(yp,t,y); 
    return;
}

/* LocalWords:  yp maxlhs
 */
