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
#include "linops/finite_diff.h"
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

#define NEW_OP_POINTER  LinopsVec[LinopCounter]

#define ADD_OP Sop = linop_chain(Sop,LinopsVec[LinopCounter]);md_copy_dims(DIMS, CurDims, linop_codomain(LinopsVec[LinopCounter])->dims);debug_printf(DP_INFO,"OutDims: ");debug_print_dims(DP_INFO,DIMS,CurDims);LinopCounter++;

complex float* dataFiles[MAX_FILES];
long Fdims[MAX_FILES][DIMS];

const struct linop_s* LinopsVec[MAX_LINOPS];
long LinopCounter=0;

const struct linop_s* LinopsOutVec[MAX_LINOPS];
long LinopOutCounter=0;

long getLinopOutCounter() { return LinopOutCounter+1; }
const struct linop_s** getLinopsOutVec() { return LinopsOutVec; }

long * getFdims(long i) { return Fdims[i]; }
complex float* getDataFile(long i) { return dataFiles[i]; }

void FreeOutLinops() {
    // debug_printf(DP_INFO,"Freeing %ld Out linops\n",LinopOutCounter);
    for(long i=0;i<LinopOutCounter;i++) {
        linop_free(LinopsOutVec[i]);
    }
    // debug_printf(DP_INFO,"FreeOutLinops done\n");
}

void FreeLinops() {
    // debug_printf(DP_INFO,"Freeing %ld linops\n",LinopCounter);
    for(long i=0;i<LinopCounter;i++) {
        linop_free(LinopsVec[i]);
    }
    // debug_printf(DP_INFO,"FreeLinops done\n");
}
void ReadScriptFiles(char* argv[],long n) {
    long i;
    debug_printf(DP_INFO,"Reading files\n");
    for(i=0;i<n;i++) {        
        dataFiles[i] = load_cfl(argv[i], DIMS, Fdims[i]);
        // printf("Reading %s: %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld %ld\n",argv[i],Fdims[i][0],Fdims[i][1],Fdims[i][2],Fdims[i][3],Fdims[i][4],Fdims[i][5],Fdims[i][6],Fdims[i][7],Fdims[i][8],Fdims[i][9],Fdims[i][10],Fdims[i][11],Fdims[i][12],Fdims[i][13],Fdims[i][14],Fdims[i][15]);
    }
    debug_printf(DP_INFO,"Finished reading files\n");
}

void ClearReadScriptFiles( char* argv[],long n) {
    long i;
    debug_printf(DP_INFO,"Clearing files' memory\n");
    for(i=0;i<n;i++) {
        // debug_printf(DP_INFO,"Clearing %s\n",argv[i]);
        unmap_cfl(DIMS, Fdims[i], dataFiles[i]);
    }
    debug_printf(DP_INFO,"Finished Clearing files' memory\n");
}

const struct linop_s* getLinopScriptFromFile(const char *FN, long CurDims[]) {
    LinopOutCounter=0;
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
    
    fp = fopen(FN, "r");
	if (fp == NULL) {
        debug_printf(DP_ERROR, "Couldn't open script file!!!");
        exit(EXIT_FAILURE); }
    
    bool NormalOp=false;

	while ((read = getline(&line, &len, fp)) != -1) {
		if(read>0) {
			if(line[0]=='#') {
				debug_printf(DP_INFO,"%s", line);
                continue;
			}
            
            for(int i = 0; line[i]; i++) { line[i] = tolower(line[i]);  }
			token = strtok(line, " ,.-\n");
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
            if(strcmp(token,"grad")==0) {
				token = strtok(NULL, " ,.-");
                long GradFlags=atoi(token);
                debug_printf(DP_INFO,"Linop %ld: Adding: grad with flag %ld\n",LinopCounter,GradFlags);
                
                NEW_OP_POINTER = linop_gradm_create(DIMS, CurDims, GradFlags);
                // debug_printf(DP_INFO,"SOP %d %d \n",linop_domain(Sop)->N,linop_codomain(Sop)->N);
                // debug_printf(DP_INFO,"Cur %d %d \n",linop_domain(LinopsVec[LinopCounter])->N,linop_codomain(LinopsVec[LinopCounter])->N);
                
                ADD_OP
			}
            if(strcmp(token,"finitediff")==0) {
				token = strtok(NULL, " ,.-");
                long FDFlags=atoi(token);
                token = strtok(NULL, " ,.");
                long bSnip=atoi(token);

                debug_printf(DP_INFO,"Linop %ld: Adding: finitediff with flag %ld snip %ld\n",LinopCounter,FDFlags, bSnip);
                
                NEW_OP_POINTER = linop_finitediff_create(DIMS, CurDims, FDFlags, bSnip);
                ADD_OP
			}
            if(strcmp(token,"zfinitediff")==0) {
				token = strtok(NULL, " ,.-");
                long zFDFlags=atoi(token);
                token = strtok(NULL, " ,.");
                long bCircular=atoi(token);

                debug_printf(DP_INFO,"Linop %ld: Adding: finitediff with flag %ld circular %ld\n",LinopCounter,zFDFlags, bCircular);
                
                NEW_OP_POINTER = linop_zfinitediff_create(DIMS, CurDims, zFDFlags, bCircular);
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
                debug_print_dims(DP_INFO,DIMS,linop_domain(LinopsVec[Idx])->dims);
                debug_print_dims(DP_INFO,DIMS,linop_codomain(LinopsVec[Idx])->dims);
                
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
            if(strcmp(token,"nextlinop")==0) {
                debug_printf(DP_INFO,"---------\nMoving to next linop\n----------------\n");
                if(NormalOp) {
                    const struct linop_s* tmp=Sop;
                    Sop=linop_PutFowrardOfBInNormalOfA(SopBackup, tmp);
                    linop_free(tmp);
                }
                LinopsOutVec[LinopOutCounter]=Sop;
                LinopOutCounter++;
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

    LinopsOutVec[LinopOutCounter]=Sop;

    FreeLinops();
    debug_printf(DP_INFO,"getLinopScriptFromFile end\n");
    return LinopsOutVec[0];
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
        "resize FileIdx : File contains new size -- not tested\n"
        "dblsz/halfsz/dblszc/halfszc Flags : Flags for dims to double/half\n"
        "wavelet xflags randshift : randshift BOOL 0/1\n"
        "NUFFT TrajFileIdx WeightsFileIdx BasisFileIdx NUFlags ToepBool pcycleBool periodicBool lowmemBool : TrajFile should be [3 readout spokes]. Bool 0/1. NUFT defaults are false.\n"
        "GRAD <Grad_flags> : *Note*: Grad results goes to the 16th dim, not BART's usual 17th\n"
        "finitediff <FD_flags> <boolean snip 0/1>\n"
        "zfinitediff <zFD_flags> <boolean circular 0/1>";

int main_linopScript(int argc, char* argv[])
{
    LinopCounter=0;

	bool Normal = false;
    bool Adj = false;
    unsigned int fftmod_flags = 0;

    unsigned int WhichLinopToRun = 0;
    unsigned int nIters = 1;

	const struct opt_s opts[] = {
		OPT_SET('N', &Normal, "Apply normal"),
        OPT_SET('A', &Adj, "Apply adjoint"),
        //OPT_UINT('j', &fftmod_flags, "fftmod_flags", "flags for fftmod_flags of (k-space?) input (if forward/normal) and (sensitivity?) file0"),
        OPT_UINT('L', &WhichLinopToRun, "idx", "Which Linop to run"),
        OPT_UINT('n', &nIters, "iters", "# times to repeat the linop (For timing tests)"),
	};	

	num_init();
    
    cmdline(&argc, argv, 0, 100, usage_str, help_str, ARRAY_SIZE(opts), opts);
	int num_args = argc - 1;
    
    debug_printf(DP_INFO,"main_linopScript\n");
    
    ReadScriptFiles(&argv[4],num_args-4);
    
    // fftmod(DIMS, getFdims(1), fftmod_flags, getDataFile(1), getDataFile(1));
    
    debug_printf(DP_INFO,"input dims: %s\n",argv[3]);
    debug_printf(DP_INFO,"XXXXXXXXXXXXXX\n");
    long inputDims_dims[DIMS];
    long input_dims[DIMS];
    complex float* inputDims = load_cfl(argv[2], DIMS, inputDims_dims);
    
    debug_printf(DP_INFO,"inputDims_dims: ");
    debug_print_dims(DP_INFO,DIMS,inputDims_dims);
    
    md_copy_dims(DIMS, input_dims, inputDims_dims);
    for(long d=0;d<inputDims_dims[0]*inputDims_dims[1];d++) {
        input_dims[d]=inputDims[d];
    }
    debug_printf(DP_INFO,"input_dims: ");
    debug_print_dims(DP_INFO,DIMS,input_dims);
        
    long inputF_dims[DIMS];
    complex float* input=load_cfl(argv[3], DIMS, inputF_dims);

    debug_printf(DP_INFO,"inputF_dims: ");
    debug_print_dims(DP_INFO,DIMS,inputF_dims);

    long CurDims[DIMS];
	md_copy_dims(DIMS, CurDims, input_dims);
    
    debug_printf(DP_INFO,"Reading script:\n");
    getLinopScriptFromFile(argv[1],CurDims);
    const struct linop_s* Sop = LinopsOutVec[WhichLinopToRun];
    
    /* Ops:
     * v FFT FFT_FLAGS
     * v IFFT FFT_FLAGS
     * v FMAC SQUASH_FLAGS             : md_merge_dims(N, dims, dims1, dims2); md_select_dims(N, ~squash, dimso, dims);
     * v identity
     * v Transpose                       ? void md_copy2(unsigned int D, const long dim[D], const long ostr[D], void* optr, const long istr[D], const void* iptr, size_t size)
     * v sampling
     * v nufft
     * v resize : pad and crop
     * v GRAD
     * v finitediff
     * v zfinitediff
     * 
     * Sum                          : fmax with singleton. Needs flags to squash
     * Permute
     * matrix
     * realval
     * wavelet
     * cdiag static struct linop_s* linop_gdiag_create(unsigned int N, const long dims[N], unsigned int flags, const complex float* diag, bool rdiag)
     * rdiag
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
        for(long i=0;i<nIters;i++) {
            linop_forward(Sop, DIMS, CurDims, out,	DIMS, input_dims, input); }
    } else if(!Normal) { // adjoint
        debug_printf(DP_INFO,"Applying the operator - adjoint\n");
        out = create_cfl(argv[num_args], DIMS, input_dims);
        for(long i=0;i<nIters;i++) {
            linop_adjoint(Sop, DIMS, input_dims, out,	DIMS, inputF_dims, input); }
    } else { // Normal
        debug_printf(DP_INFO,"Applying the operator - Normal\n");
        out = create_cfl(argv[num_args], DIMS, input_dims);
        for(long i=0;i<nIters;i++) {
            linop_normal(Sop, DIMS, input_dims, out, input); }
    }
   
    debug_printf(DP_INFO,"Saving output\n");
    if(!Adj) {
        unmap_cfl(DIMS, linop_codomain(Sop)->dims, out);
    } else {
        unmap_cfl(DIMS, linop_domain(Sop)->dims, out);
    }
    
    debug_printf(DP_INFO,"inputF_dims: ");
    debug_print_dims(DP_INFO,DIMS,inputF_dims);

    unmap_cfl(DIMS, inputF_dims, input);
    
    ClearReadScriptFiles(&argv[4],num_args-4);

    FreeOutLinops();
    
	exit(EXIT_SUCCESS);

	return 0;
}

