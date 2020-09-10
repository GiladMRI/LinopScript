/* Copyright 2013-2018. The Regents of the University of California.
 * Copyright 2015-2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012-2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 * 2014-2016 Frank Ong <frankong@berkeley.edu>
 * 2014-2018 Jon Tamir <jtamir@eecs.berkeley.edu>
 *
 */

#include <assert.h>
#include <stdbool.h>
#include <complex.h>
#include <stdio.h>
#include <math.h>
#include <time.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/init.h"
#include "num/ops.h"

#include "iter/misc.h"
#include "iter/italgos.h"
#include "iter/iter.h"
#include "iter/iter2.h"
#include "iter/prox.h"

#include "linops/linop.h"
#include "linops/linopScript.h"
#include "linops/fmac.h"
#include "linops/sampling.h"
#include "linops/someops.h"

#include "noncart/nufft.h"

#include "sense/recon.h"
#include "sense/model.h"
#include "sense/optcom.h"

#include "misc/debug.h"
#include "misc/mri.h"
#include "misc/utils.h"
#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"

#include "grecon/optreg.h"
#include "grecon/italgo.h"
#include "lowrank/lrthresh.h"


#include "num/iovec.h"
#include "num/ops.h"
#include "num/rand.h"

#include "wavelet/wavthresh.h"

#include "misc/png.h"

static const char usage_str[] = "<OpScriptTxt> <StartDims> <kspace> [<file0> [<file> [<file2> [...]]]] <output>";
static const char help_str[] = "inexact proximal splitting - linop script.";

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


static float frand()  //  uniform distribution, (0..1] 
{
  return (rand()+1.0)/(RAND_MAX+1.0);
}

#ifndef MAX_FN_LEN
#define MAX_FN_LEN 1024
#endif
// #define PRINT_TO_FNBUF(X) if (1024 <= snprintf(FNbuf, MAX_FN_LEN, "%s%s", FNbase, "ElemsAlpha")) error("Error snprintf %s", "ElemsAlpha");
#define PRINT_TO_FNBUF(X) if (1024 <= snprintf(FNbuf, MAX_FN_LEN, "%s%s", FNbase, X)) error("Error snprintf %s", X);
#define PRINT_TO_FNBUFOUT(X) if (1024 <= snprintf(FNbuf, MAX_FN_LEN, "%s%s", FNbaseOut, X)) error("Error snprintf %s", X);
#define PRINT_TO_FNBUFWS(X) if (1024 <= snprintf(FNbuf, MAX_FN_LEN, "%s%s", WSFNBase, X)) error("Error snprintf %s", X);
const long MAX_MAPS=10;
const long N_AUX_MAPS=3;  // Only for T currently
const long N_RAND_MAPS_FOR_P=10;
const long ERR_VEC_LENGTH=1000000;


void CalcOthers(unsigned int D, const long xdims[D], complex float* others, long nelements, long curelem, complex float* sluelems[MAX_MAPS],
    bool translation, struct operator_s* TrasOp1)
{
    debug_printf(DP_DEBUG3, "Calculating 'Others' %d\n",curelem);

    md_clear(D, xdims, others, CFL_SIZE);
    md_zexpj(D, xdims, others, others);

    // cur_time1=  timestamp();
    for(long j=0;j<nelements;j++) {
        // if(translation && (j==nelements-1) ) {
        //     operator_apply(TrasOp1,D, xdims, others, D, xdims, others); }
        if(j!=curelem) {
            md_zmul(D, xdims, others, others, sluelems[j]);
        }
    }
    // if(Translation) {
    //                     operator_apply(XF2DOp->adjoint,N, XDims, r, N, XDims, r); }
}

// f is data consistency term, g is regularization term
// QL(x,y)=f(y)+<x-y,grad-f(y)>+L/2 ||x-y||^2 + g(x)
// New=prox_t(y-t*grad-f(y))
// while f(New)>f(Old)+ <x-y,grad-f(y)> + 1/(2*t) * ||x-y||^2
// t=beta*t
// New=prox_t(y-t*grad-f(y))
// end
void OneLineSearchIteration(const struct operator_s* ForwardOp, unsigned int D, const long dims[D],
    const complex float* Gradf, float t0, float beta, const complex float* sig, const complex float* TmpSigDomain,
    complex float* trgMap, complex float* tmpElem, const complex float* srcMap,
    const long dimsR[D], long CurEType, float CurL, struct iter_op_p_s CurProxOp,
    bool DoWrappingTrick, const complex float* ElemMin, const complex float* ElemMax,
    const complex float* ThisElemRandMaps[N_RAND_MAPS_FOR_P], float *pRelMapChangeWithAlphaAfterProx) 
    // ApplyProximal(unsigned int D, const long mapDims[D], const long mapDimsR[D],
    // complex float* trgMap, complex float* tmpMap, const complex float* srcMap,
    // long CurEType, float CurA, float CurL, struct iter_op_p_s CurProxOp,
    // bool DoWrappingTrick, const complex float* ElemMin, const complex float* ElemMax,
    // const complex float* ThisElemRandMaps[N_RAND_MAPS_FOR_P], float *pRelMapChangeWithAlphaAfterProx) 
{
    float t=t0;
    float fOld,fNew,fDistSquare, fDotProd;
    // apply ForwardOp on srcMap into TmpSigDomain
    operator_apply(ForwardOp,D, operator_codomain(ForwardOp)->dims, TmpSigDomain, D, dims, srcMap);
    // TmpSigDomain=TmpSigDomain-sig
    md_zsub(D, dims, TmpSigDomain, TmpSigDomain, sig); // optr = iptr1 - iptr2
    
    // fOld=sum square TmpSigDomain
    fOld=md_zscalar_real(D, operator_codomain(ForwardOp)->dims, TmpSigDomain, TmpSigDomain);
    
    // New=prox_t(y-t*grad-f(y))
    // ApplyProximal on srcMap to trgMap
    float RelMapChangeWithAlphaAfterProx;
    ApplyProximal(D, dims, dimsR,
        trgMap, tmpElem, srcMap,
        CurEType, t, CurL, CurProxOp,
        DoWrappingTrick, ElemMin, ElemMax, ThisElemRandMaps, &RelMapChangeWithAlphaAfterProx);
    // apply ForwardOp on trgMap into TmpSigDomain
    operator_apply(ForwardOp,D, operator_codomain(ForwardOp)->dims, TmpSigDomain, D, dims, trgMap);
    // fNew=sum square TmpSigDomain
    fNew=md_zscalar_real(D, operator_codomain(ForwardOp)->dims, TmpSigDomain, TmpSigDomain);
    // set tmpElem to x-y
    md_zsub(D, dims, tmpElem, trgMap, srcMap); // optr = iptr1 - iptr2
    // calc ||x-y||^2
    fDistSquare=md_zscalar_real(D, dims, tmpElem, tmpElem);
    // calc <x-y,grad-f(y)>
    fDotProd=crealf(md_zscalar(D,dims,tmpElem, Gradf))/(2.0f*t);

    // while f(New)>f(Old)+ <x-y,grad-f(y)> + 1/(2*t) * ||x-y||^2
    while(fNew> (fOld+fDotProd+fDistSquare)) {
        // t=beta*t
        t*=beta;
        // New=prox_t(y-t*grad-f(y))
        ApplyProximal(D, dims, dimsR,
        trgMap, tmpElem, srcMap,
        CurEType, t, CurL, CurProxOp,
        DoWrappingTrick, ElemMin, ElemMax, ThisElemRandMaps, &RelMapChangeWithAlphaAfterProx);
        // apply ForwardOp on trgMap into TmpSigDomain
        operator_apply(ForwardOp,D, operator_codomain(ForwardOp)->dims, TmpSigDomain, D, dims, trgMap);
        // fNew=sum square TmpSigDomain
        fNew=md_zscalar_real(D, operator_codomain(ForwardOp)->dims, TmpSigDomain, TmpSigDomain);
        // set tmpElem to x-y
        md_zsub(D, dims, tmpElem, trgMap, srcMap); // optr = iptr1 - iptr2
        // calc ||x-y||^2
        fDistSquare=md_zscalar_real(D, dims, tmpElem, tmpElem);
        // calc <x-y,grad-f(y)>
        fDotProd=md_zscalar(D,dims,tmpElem, Gradf)/(2*t);
    }
}


// Power method
float GetMaxEigPowerMethod(const struct operator_s* op, unsigned int D, const long dims[D],
    complex float* Elem, complex float* pElem,
    long MaxLIters, float ErrRMSRatioT, long *p_nIters,float RandFacToAdd)
{
    debug_printf(DP_DEBUG3, "Creating rand\n");
    // md_gaussian_rand(D, dims, pElem);
    if(RandFacToAdd>1e-10) {
        md_gaussian_rand(D, dims, Elem);
        md_zsmul(D, dims, Elem, Elem, RandFacToAdd);
        // md_zadd(D, dims, pElem, Elem, pElem);
        md_copy(D, dims, pElem, Elem, CFL_SIZE);
    }

    float L,ErrRMSRatio,CurRMS,CurRMSAx,nom_ir,denom_ir,NewAlpha;
    // complex float denom;
    // complex float nom;
    float nom,denom;
    // float ErrRMSRatioT=1e-2;
    
    // CurRMS=md_zrms(D, dims, pElem);
    // debug_printf(DP_DEBUG3, "RandRMS=%e\n",CurRMS);
    long i;

    debug_printf(DP_DEBUG3, "GetMaxEigPowerMethod running iters\n");

    bool SmulOnCPU=false;
    complex float *tmpCPU=NULL;
    for (i = 0; i < MaxLIters; i++) {
        // debug_printf(DP_INFO, "Power iter %d start\n",i);
        // one iter
        operator_apply(op,D, dims, Elem, D, dims, pElem);
        // md_copy(D, dims, Elem, pElem, CFL_SIZE);
        // now check how the factor and err rms
        // [a/b a*conj(b.')/(norm(b).^2) a*conj(b.')/sum(b*conj(b.')) ]
        // denom=md_zscalar(D, dims, pElem, pElem);
        // nom=md_zscalar(D, dims, Elem, pElem);
        denom=md_zscalar_real(D, dims, pElem, pElem);
        nom=md_zscalar_real(D, dims, Elem, pElem);
        // denom=nom;
        // nom=denom;
        // denom=1.0f;
        // nom=1.0f;
        
        // nom_ir=cimagf(nom)/crealf(nom);
        // denom_ir=cimagf(denom)/crealf(denom);
        // L=crealf(nom)/crealf(denom);
        L=nom/denom;
        
        md_zsmul(D, dims, pElem, pElem, L);
        md_zsub(D, dims, pElem, Elem, pElem); // optr = iptr1 - iptr2
        
        CurRMS=md_zrms(D, dims, pElem);
        CurRMSAx=md_zrms(D, dims, Elem);
        // if(nom_ir>1e-5 || denom_ir>1e-5) { debug_printf(DP_INFO, "Nom,Denom i/r %6.2e %6.2e\n",nom_ir,denom_ir); }
        debug_printf(DP_DEBUG3, "CurRMS=%6.2e, CurRMSAx=%6.2e, nom=%6.2e denom=%6.2e\n",CurRMS,CurRMSAx,nom,denom);
        ErrRMSRatio=CurRMS/CurRMSAx;

        md_copy(D, dims, pElem, Elem, CFL_SIZE);
        
        if(ErrRMSRatio<ErrRMSRatioT) break;

        float Fac=(1.0f/CurRMSAx);
        // if(fabs(CurRMSAx)>1e3 || fabs(CurRMSAx)<1e-3) {
        if(!SmulOnCPU) {
            md_zsmul(D, dims, pElem, pElem, Fac);

            if(md_zrms(D, dims, pElem)<1e-5) {
                debug_printf(DP_INFO, "ERRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRRR\n");
                error("Power iter lambda calculation error, try changing linear op scale\n");
                SmulOnCPU=true;
                tmpCPU = md_alloc(DIMS, dims, CFL_SIZE);
            }
        }
        if(SmulOnCPU) {
            md_copy(D, dims, tmpCPU, Elem, CFL_SIZE);
            md_zsmul(D, dims, tmpCPU, tmpCPU, Fac);
            md_copy(D, dims, pElem, tmpCPU, CFL_SIZE);
            // md_gaussian_rand(D, dims, pElem);
        }
        debug_printf(DP_DEBUG4, "BBB %e %e %e\n",md_zrms(D, dims, Elem),md_zrms(D, dims, pElem),Fac);
        
        // end one iter
        debug_printf(DP_DEBUG3, "Power iter %3d L=%6.2e Rerr=%6.2e\n",i,L,ErrRMSRatio);
    }
    if(SmulOnCPU) { md_free(tmpCPU); }
    debug_printf(DP_DEBUG1, "Power iter %3d L=%6.2e Rerr=%6.2e\n",i,L,ErrRMSRatio);
    (*p_nIters)=i+1;
    md_zsmul(D, dims, pElem, pElem, 1.0f/CurRMSAx);
    return L;
}



/*ApplyProximal(N, mapsDims[curElem], mapsDimsR[curElem],
    Elems[curElem], pElems[curElem], tmpElems[curElem],
    CurElementType, creal(CurAlpha),creal(ElemsLambda[curElem]), it_prox_ops[curElem],
    WrappingTrick[curElem], ElemsMin[curElem], ElemsMax[curElem],RandMaps[curElem],&RelMapChangeWithAlphaAfterProx);
*/
// Applies proximal on srcMap into trgMap, using tmpMap as temporary
void ApplyProximal(unsigned int D, const long mapDims[D], const long mapDimsR[D],
    complex float* trgMap, complex float* tmpMap, const complex float* srcMap,
    long CurEType, float CurA, float CurL, struct iter_op_p_s CurProxOp,
    bool DoWrappingTrick, const complex float* ElemMin, const complex float* ElemMax,
    const complex float* ThisElemRandMaps[N_RAND_MAPS_FOR_P], float *pRelMapChangeWithAlphaAfterProx) 
{
    debug_printf(DP_DEBUG3, "Applying proximal\n");
    // Apply proximal: Elems[curElem] -> Elems[curElem]
    if(DoWrappingTrick && (CurL>0) ) { // wrapping stuff for P
        debug_printf(DP_DEBUG2, "WrappingTrick\n");
        long rndI=floor(frand()*N_RAND_MAPS_FOR_P);

        md_clear(D, mapDims, tmpMap, CFL_SIZE);
        md_zadd(D, mapDims, tmpMap, tmpMap, ThisElemRandMaps[rndI]);
        
        
        md_zexpj(D, mapDims,tmpMap,tmpMap);

        md_zexpj(D, mapDims,trgMap,srcMap);
        md_zmul(D, mapDims,trgMap,trgMap,tmpMap);
        md_zarg(D, mapDims,tmpMap,trgMap);                   

        // Here calling prox
        if(CurL>0) {
            iter_op_p_call(CurProxOp, CurA, (float*)trgMap, (float*)tmpMap);
        } else {
            md_copy(D, mapDims, trgMap, tmpMap, CFL_SIZE);
        }

        md_clear(D, mapDims, tmpMap, CFL_SIZE);
        md_zsub(D, mapDims, tmpMap, tmpMap, ThisElemRandMaps[rndI]);

        md_zexpj(D, mapDims,tmpMap,tmpMap);
        md_zexpj(D, mapDims,trgMap,trgMap);
        md_zmul(D, mapDims,trgMap,trgMap,tmpMap);
        md_zarg(D, mapDims,trgMap,trgMap);
        debug_printf(DP_DEBUG2, "ok WrappingTrick\n");
    } else { // No wrapping trick
        if(CurL>0) {
            iter_op_p_call(CurProxOp, CurA, (float*)trgMap, (float*)srcMap);
        } else { // No proximal
            md_copy(DIMS, mapDims, trgMap, srcMap, CFL_SIZE); // ok
        }
    }
    if(CurEType==1) { // abs for M
        md_zabs(D, mapDims, trgMap, trgMap);
    }
    if(CurEType==4) { // max with 5 for T2*
        debug_printf(DP_DEBUG3, "Max op\n");
        md_zabs(D, mapDims, trgMap, trgMap);
    }
    if(CurEType!=5) {
        md_min(D, mapDimsR, trgMap, ElemMax, trgMap);
        md_max(D, mapDimsR, trgMap, ElemMin, trgMap);
    }

    md_zsub(D, mapDims, tmpMap, srcMap, trgMap);
    (*pRelMapChangeWithAlphaAfterProx)=md_zrms(D, mapDims, tmpMap)/md_zrms(D, mapDims, trgMap);
}










int main_splitProx(int argc, char* argv[])
{
    // Initialize default parameters
    long maxiter = 5;
    long randshift=1;
    bool dohogwild=false;
    long TimeBetweenSaves=1000000;
    bool hasGT=false;

    char FNbuf[MAX_FN_LEN];
    char FNbufx[MAX_FN_LEN];

    const char* FNbase = NULL;
    const char* FNbaseOut = NULL;

    const char* SigFN = NULL;
    const char* WSFNBase = NULL;

    const char* Msg = NULL;

    bool gpu = false;
    unsigned int gpun = 0;
    long UseOS = 0;
    bool Force=false;
    float tol=0.9999f;
    long iter_to_tol=500;

    long IntermediateSaveIter=0;
    // bool Calc_r_once_per_iter=false;
    long Calc_r_once_per_iter_StartIter=1000000;

    bool UseLineSearch=false;
    bool UseISTALineSearch=false;
    bool UseLipshitz=false;

    unsigned int llr_blk = 8;

    long xflags=3; // 7;
    long SquashFlags=0;
    long jflags=0;
    long Ljflags=0;
    //long SquashFlags=64;

    float ItersWithoutLineSearchAsRatioOfIter=0.1f;
    long MaxPowerIters=20;

    bool Translation=false;

    const struct opt_s opts[] = {
        OPT_LONG('i', &maxiter, "iter", "max. number of iterations"),
        OPT_LONG('s', &TimeBetweenSaves, "TimeBetweenSaves", "Save output every x seconds"),
        OPT_LONG('q', &SquashFlags, "SquashFlags", "Flags of dims to squash on maps' fmac"),
        OPT_LONG('D', &xflags, "Reg Flags", "Flags of dims to regularize on (default 3)"),
        OPT_LONG('j', &jflags, "Joint Flags", "Flags of dims to jointly regularize"),
        OPT_LONG('c', &Ljflags, "Joint Flags for LLR", "Flags of dims to jointly regularize in LLR"),
        OPT_UINT('b', &llr_blk, "blk", "Lowrank block size"),
        // OPT_UINT('n', &randshift, "disable random wavelet cycle spinning"),
        OPT_SET('1', &Translation, "Translation"),
        OPT_SET('H', &dohogwild, "(hogwild)"),
        OPT_SET('t', &hasGT, "Use Ground truth"),
        OPT_SET('L', &UseLineSearch, "Use Line search"),
        OPT_SET('A', &UseISTALineSearch, "Use ISTA Line search"),
        OPT_SET('P', &UseLipshitz, "Use Lipshitz estimation with power method"),
        // OPT_SET('p', &Calc_r_once_per_iter, "Calc_r_once_per_iter"),
        OPT_LONG('p', &Calc_r_once_per_iter_StartIter, "Calc_r_once_per_iter_StartIter","Calc_r_once_per_iter_StartIter"),
        OPT_STRING('F', &FNbase, "FNbase", "file names prefix"),
        OPT_STRING('O', &FNbaseOut, "FNbaseOut", "file names prefix for output"),
        OPT_STRING('W', &WSFNBase, "WSFNBase", "file names prefix for warm start maps"),
        OPT_STRING('S', &SigFN, "SigFN", "file name A^H*signal"),
        OPT_STRING('m', &Msg, "Msg", "Message for iter prints"),
        OPT_SET('f', &Force, "Force calculation even if maps file already exists"),
        OPT_SET('g', &gpu, "use GPU"),
        OPT_UINT('G', &gpun, "gpun", "use GPU device gpun"),
        OPT_INT('d', &debug_level, "level", "Debug level"),
        OPT_LONG('o', &UseOS, "pixels","in-plane Oversampling value"),
        OPT_LONG('I', &IntermediateSaveIter, "iterations","save maps every x iterations"),
        OPT_FLOAT('l',&tol,"tol value","tol value (default 0.9999"),
        OPT_LONG('x', &MaxPowerIters, "MaxPowerIters","Max Power Iters"),
        OPT_FLOAT('r',&ItersWithoutLineSearchAsRatioOfIter,"ItersWithoutLineSearchAsRatioOfIter","Iters without LineSearch As Ratio Of Iter (default 0.1"),
        OPT_LONG('T',&iter_to_tol,"iter_to_tol","iteration to go back for tol comparison (deafult 500)")
    };

    cmdline(&argc, argv, 0, 100, usage_str, help_str, ARRAY_SIZE(opts), opts);

    // Linop script
    int num_args = argc - 1;


    if (gpu) {
        num_init_gpu_device(gpun);    }
    else {
        num_init();    }

    if (gpu) {
#ifdef USE_CUDA
        debug_printf(DP_INFO, "GPU reconstruction\n");    
#else
        debug_printf(DP_INFO, "no USE_CUDA\n");    
#endif
    }

    if(UseOS>0) {
        debug_printf(DP_INFO, "Using oversampling %ld\n",UseOS);
    }
    
    if(Translation) {
        debug_printf(DP_INFO, "Using Translation\n");   
    }

    debug_printf(DP_INFO, "main_splitProx\n");
    
    // if(Calc_r_once_per_iter) {
    //     debug_printf(DP_INFO, "Using Calc_r_once_per_iter\n");
    // }
    // bool Calc_r_every_time=!Calc_r_once_per_iter;
    if(FNbase==NULL) {
        error("Need FNbase, use -F\n");    }

    if(WSFNBase==NULL) {
        WSFNBase=FNbase;
    }

    if(FNbaseOut==NULL) {
        FNbaseOut=FNbase;
    }
    PRINT_TO_FNBUFOUT("Elem0.cfl");

    char DefMsg[]=" ";
    if(Msg==NULL) {
        Msg=DefMsg;
    }

    debug_printf(DP_INFO, "Checking %s\n",FNbuf);

    FILE *file;
    bool MapsExist=false;
    file = fopen(FNbuf, "r");
    if(file!=NULL) {
        MapsExist=true;
        fclose(file);
        if(!Force) {
            printf("maps file already exists, skipping!\n");
            return 0;
        }
    }

    const complex float One=1;
   
    long i,j;
    long CurElementType;
   
    int N = DIMS;

    long nElements;

    long mapsDims[MAX_MAPS][N];
    long mapsDimsR[MAX_MAPS][N];

    long mapsDimsMin[MAX_MAPS][N];
    long mapsDimsMax[MAX_MAPS][N];

    long mapsStrs[MAX_MAPS][N];
    long LDims[MAX_MAPS][N];
    long LStrs[MAX_MAPS][N];
    complex float* Elems0[MAX_MAPS];
    complex float* ElemsLa[MAX_MAPS];
    complex float* ElemsL[MAX_MAPS];

    complex float* ElemsMin[MAX_MAPS];
    complex float* ElemsMax[MAX_MAPS];

    long ElemsDataDims[N];
    PRINT_TO_FNBUF("ElemsLambda");
    complex float* ElemsLambda = load_cfl(FNbuf, N, ElemsDataDims);
    nElements=ElemsDataDims[0];
    debug_printf(DP_INFO,"Using %ld Elements\n",nElements);

    complex float* ElemsAlpha;
    if(UseLipshitz) {
        ElemsAlpha=md_alloc(N, ElemsDataDims, CFL_SIZE);
        md_clear(N, ElemsDataDims, ElemsAlpha, CFL_SIZE);
    } else {
        PRINT_TO_FNBUF("ElemsAlpha");
        ElemsAlpha=load_cfl(FNbuf, N, ElemsDataDims);    
    }
    
    PRINT_TO_FNBUF("ElementTypes");
    complex float* ElementTypes = load_cfl(FNbuf, N, ElemsDataDims);

    // int ret = sscanf(optarg, "%*[^:]:%f:%f:%f:%f:%f:%f:%f:%f:%f:%f:%f:%f", &regs[r].xflags, &regs[r].jflags, &regs[r].k);
    //         assert(3 == ret);


    PRINT_TO_FNBUF("ninneriter");
    complex float* ninneriter=load_cfl(FNbuf, N, ElemsDataDims);

    long ProxTypes[MAX_MAPS];
    for(i=0;i<nElements;i++) {
        ProxTypes[i]=round(ElementTypes[i])/100;
        ElementTypes[i]=((long) round(ElementTypes[i]))%100;
    }

    PRINT_TO_FNBUFOUT("Summary.txt");

    FILE *sfile;
    sfile = fopen(FNbuf, "wt");
    if(sfile!=NULL) {
        fprintf(sfile,"-- summary --\n");
        fprintf(sfile,"Alpha:  ");
        for(i=0;i<nElements;i++) {
            fprintf(sfile,"%20f ",creal(ElemsAlpha[i]));
        }
        fprintf(sfile,"\nLambda: ");
        for(i=0;i<nElements;i++) {
            fprintf(sfile,"%20f ",creal(ElemsLambda[i]));
        }
        fprintf(sfile,"\niter:   ");
        for(i=0;i<nElements;i++) {
            fprintf(sfile,"%d ",round(creal(ninneriter[i])));
        }
        fprintf(sfile,"\n----------\n");
        fclose(sfile);
    }

    for(i=0;i<nElements;i++) {
        snprintf(FNbufx, MAX_FN_LEN, "ElemsWS_%ld", i);
        PRINT_TO_FNBUFWS(FNbufx);
        Elems0[i]=load_cfl(FNbuf, N, mapsDims[i]);

        md_copy_dims(DIMS, mapsDimsR[i], mapsDims[i]);
        mapsDimsR[i][0]=mapsDims[i][0]*2;

        snprintf(FNbufx, MAX_FN_LEN, "ElemsMin_%ld", i);
        PRINT_TO_FNBUFWS(FNbufx);
        ElemsMin[i]=load_cfl(FNbuf, N, mapsDimsMin[i]);

        snprintf(FNbufx, MAX_FN_LEN, "ElemsMax_%ld", i);
        PRINT_TO_FNBUFWS(FNbufx);
        ElemsMax[i]=load_cfl(FNbuf, N, mapsDimsMax[i]);

        md_calc_strides(DIMS, mapsStrs[i], mapsDims[i], CFL_SIZE);
        snprintf(FNbufx, MAX_FN_LEN, "ElemsL_%ld", i);
        PRINT_TO_FNBUF(FNbufx);
        ElemsLa[i]=load_cfl(FNbuf, N, LDims[i]);
        md_calc_strides(DIMS, LStrs[i], LDims[i], CFL_SIZE);
        if(gpu) {
#ifdef USE_CUDA
            ElemsL[i]=md_gpu_move(DIMS, LDims[i], ElemsLa[i], CFL_SIZE);
#endif
        } else {
            ElemsL[i]=ElemsLa[i];
        }
    }

    print_cuda_meminfo();

    complex float* Elemsa[MAX_MAPS];

    complex float* Elems[MAX_MAPS];
    complex float* duElems[MAX_MAPS];
    complex float* pElems[MAX_MAPS];
    complex float* tmpElems[MAX_MAPS];
    complex float* tmpElems2[MAX_MAPS];
    complex float* tmpMapForPowerMethod[MAX_MAPS];
    
    for(i=0;i<nElements;i++) {
        snprintf(FNbufx, MAX_FN_LEN, "Elem%ld", i);
        PRINT_TO_FNBUFOUT(FNbufx);
        Elemsa[i]=create_cfl(FNbuf, N, mapsDims[i]);
        md_copy(DIMS, mapsDims[i], Elemsa[i], Elems0[i], CFL_SIZE);

        if(gpu) {
#ifdef USE_CUDA
            Elems[i]=md_gpu_move(DIMS, mapsDims[i], Elemsa[i], CFL_SIZE);
            duElems[i] = md_alloc_gpu(DIMS, mapsDims[i], CFL_SIZE);
            pElems[i] = md_alloc_gpu(DIMS, mapsDims[i], CFL_SIZE);
            tmpElems[i] = md_alloc_gpu(DIMS, mapsDims[i], CFL_SIZE);
            tmpElems2[i] = md_alloc_gpu(DIMS, mapsDims[i], CFL_SIZE);
            tmpMapForPowerMethod[i] = md_alloc_gpu(DIMS, mapsDims[i], CFL_SIZE);
            
            ElemsMin[i]=md_gpu_move(DIMS, mapsDims[i], ElemsMin[i], CFL_SIZE);
            ElemsMax[i]=md_gpu_move(DIMS, mapsDims[i], ElemsMax[i], CFL_SIZE);
#endif
        } else {
            Elems[i]=Elemsa[i];
            duElems[i] = md_alloc(DIMS, mapsDims[i], CFL_SIZE);
            pElems[i] = md_alloc(DIMS, mapsDims[i], CFL_SIZE);
            tmpElems[i] = md_alloc(DIMS, mapsDims[i], CFL_SIZE);
            tmpElems2[i] = md_alloc(DIMS, mapsDims[i], CFL_SIZE);
            tmpMapForPowerMethod[i] = md_alloc(DIMS, mapsDims[i], CFL_SIZE);
        }
    }
    for(i=0;i<nElements;i++) { md_gaussian_rand(N, mapsDims[i], tmpMapForPowerMethod[i]); }

    long sig_adjDims[DIMS];
    complex float* sig_adja;
    if(SigFN==NULL) {
        PRINT_TO_FNBUF("sig_adj");
        sig_adja = load_cfl(FNbuf, N, sig_adjDims); 
    } else {
        sig_adja = load_cfl(SigFN, N, sig_adjDims); 
    }

    complex float* sig_adj;
    if(gpu) {
#ifdef USE_CUDA
        sig_adj=md_gpu_move(DIMS, sig_adjDims, sig_adja, CFL_SIZE);
#endif
    } else {
        sig_adj=sig_adja;
    }

    complex float* GTa;
    complex float* GT;
    float GTrms;
    if(hasGT) {
        PRINT_TO_FNBUF("GT");
        long GTDims[DIMS];
        GTa = load_cfl(FNbuf, N, GTDims);
        // GT = load_cfl("/autofs/space/daisy_002/users/Gilad/gUM/GT", N, XDims);
        GTrms=md_zrms(N, GTDims, GTa);
        debug_printf(DP_INFO, "GTrms %g\n",GTrms);

        if(gpu) {
#ifdef USE_CUDA
            GT=md_gpu_move(DIMS, GTDims, GTa, CFL_SIZE);
#endif
        } else {
            GT=GTa;
        }
    }


    debug_printf(DP_INFO, "LDims[0]:");
    debug_print_dims(DP_INFO,DIMS,LDims[0]);

    print_cuda_meminfo();

    debug_printf(DP_INFO,"read elemets WS,L\n");

    long MergedDims[DIMS];
    long NewDims[DIMS];

    long OSDims[2][N];
    complex float* OSa[2];
    complex float* OS[MAX_MAPS];
    const struct linop_s* OSLinops[2];
    const struct linop_s* OSLinop;
    if(UseOS>0) {
        /*PRINT_TO_FNBUF("OSx");
        OSa[0]=load_cfl(FNbuf, N, OSDims[0]);
        PRINT_TO_FNBUF("OSy");
        OSa[1]=load_cfl(FNbuf, N, OSDims[1]);

        for(i=0;i<2;i++) {
            if(gpu) {
    #ifdef USE_CUDA
                debug_printf(DP_INFO, "Moving OS[%d] to GPU:\n",i);
                OS[i]=md_gpu_move(DIMS, OSDims[i], OSa[i], CFL_SIZE);
    #endif
            } else {
                OS[i]=OSa[i];
            }
            if(i==0) {
                SquashFlags=1;
            }
            if(i==1) {
                SquashFlags=2;
            }
            debug_printf(DP_INFO, "OSDims[%d]:\n",i);
            debug_print_dims(DP_INFO,DIMS,OSDims[i]);

            long tmpDims[DIMS];
            if(i==0) {
                md_copy_dims(DIMS, tmpDims, mapsDims[0]);
            }
            if(i==1) {
                md_copy_dims(DIMS, tmpDims, linop_codomain(OSLinops[0])->dims);
            }

            debug_printf(DP_INFO, "tmpDims:");
            debug_print_dims(DP_INFO,DIMS,tmpDims);
            md_merge_dims(DIMS, MergedDims, tmpDims, OSDims[i]);
            long CurFlags=md_nontriv_dims(DIMS,tmpDims);
            
            md_select_dims(DIMS, ~SquashFlags, NewDims, MergedDims);
            
            debug_printf(DP_INFO, "MergedDims:");
            debug_print_dims(DP_INFO,DIMS,MergedDims);

            long NewFlags=md_nontriv_dims(DIMS,NewDims);
            long TFlags=md_nontriv_dims(DIMS,OSDims[i]);
            
            OSLinops[i] = linop_fmac_create(DIMS, MergedDims, ~NewFlags, ~CurFlags, ~TFlags, OS[i]);
            debug_printf(DP_INFO, "OSLinops[%d] created\n",i);
        }

        complex float* tmpX1 = md_alloc_gpu(DIMS, linop_codomain(OSLinops[0])->dims, CFL_SIZE);
        debug_printf(DP_INFO, "Runnning OSLinops[0]:\n");
        operator_apply(OSLinops[0]->forward,DIMS, linop_codomain(OSLinops[0])->dims, tmpX1, DIMS, mapsDims[0], Elems[0]);
        debug_printf(DP_INFO, "Finished OSLinops[0]\n");

        debug_printf(DP_INFO, "B Runnning OSLinops[0]:\n");
        operator_apply(OSLinops[0]->forward,DIMS, linop_codomain(OSLinops[0])->dims, tmpX1, DIMS, mapsDims[0], Elems[0]);
        debug_printf(DP_INFO, "B Finished OSLinops[0]\n");

        complex float* tmpX2 = md_alloc_gpu(DIMS, linop_codomain(OSLinops[1])->dims, CFL_SIZE);
        debug_printf(DP_INFO, "Runnning OSLinops[1]:\n");
        operator_apply(OSLinops[1]->forward,DIMS, linop_codomain(OSLinops[1])->dims, tmpX2, DIMS, linop_codomain(OSLinops[0])->dims, tmpX1);
        debug_printf(DP_INFO, "Finished OSLinops[1]\n");

        // return 0;


        OSLinop=linop_chain(OSLinops[0],OSLinops[1]);*/

        debug_printf(DP_INFO, "OS:\n");

        long CurDims[DIMS];
        md_copy_dims(DIMS, CurDims, mapsDims[0]);
        long OSFlags=3;
        const struct linop_s* FOp = linop_fftc_create(DIMS, CurDims, OSFlags);
        long padDims[DIMS];
        md_copy_dims(DIMS, padDims, CurDims);
        padDims[0]=UseOS;
        padDims[1]=UseOS;
        const struct linop_s* PadOp = linop_resize_create(DIMS, padDims,CurDims);
        const struct linop_s* IFOp = linop_ifftc_create(DIMS, padDims, OSFlags);

        OSLinop=linop_chain(linop_chain(FOp,PadOp),IFOp);

        debug_printf(DP_INFO, "OSLinop domain,codomain:\n");
        debug_print_dims(DP_INFO,DIMS,linop_domain(OSLinop)->dims);
        debug_print_dims(DP_INFO,DIMS,linop_codomain(OSLinop)->dims);
    } else {
        for(i=0;i<2;i++) {
            OS[i]=NULL;
            OSLinops[i]=NULL;
        }
        OSLinop=NULL;
    }

    const struct linop_s* ElemsLinops[MAX_MAPS];

    for(i=0;i<nElements;i++) {
        if(UseOS>0) {
            long mapsOSDims[DIMS];
            md_copy_dims(DIMS, mapsOSDims, linop_codomain(OSLinop)->dims);

            md_merge_dims(DIMS, MergedDims, mapsOSDims, LDims[i]);
            md_select_dims(DIMS, ~SquashFlags, NewDims, MergedDims);
            
            debug_printf(DP_INFO, "MergedDims:");
            debug_print_dims(DP_INFO,DIMS,MergedDims);

            long CurFlags=md_nontriv_dims(DIMS,mapsOSDims);
            long NewFlags=md_nontriv_dims(DIMS,NewDims);
            long TFlags=md_nontriv_dims(DIMS,LDims[i]);
            
            ElemsLinops[i] = linop_fmac_create(DIMS, MergedDims, ~NewFlags, ~CurFlags, ~TFlags, ElemsL[i]);
            debug_printf(DP_INFO, "ElemsLinops[%d] created\n",i);
            
            debug_printf(DP_INFO, "OSLinop domain,codomain:\n");
            debug_print_dims(DP_INFO,DIMS,linop_domain(OSLinop)->dims);
            debug_print_dims(DP_INFO,DIMS,linop_codomain(OSLinop)->dims);

            debug_printf(DP_INFO, "ElemsLinops[%d] domain,codomain:\n",i);
            debug_print_dims(DP_INFO,DIMS,linop_domain(ElemsLinops[i])->dims);
            debug_print_dims(DP_INFO,DIMS,linop_codomain(ElemsLinops[i])->dims);

            complex float* tmpX1 = md_alloc_gpu(DIMS, linop_codomain(OSLinop)->dims, CFL_SIZE);
            debug_printf(DP_INFO, "Runnning OSLinop:\n");
            operator_apply(OSLinop->forward,N, linop_codomain(OSLinop)->dims, tmpX1, N, mapsDims[i], Elems[i]);
            debug_printf(DP_INFO, "Finished OSLinop\n");

            complex float* tmpX2 = md_alloc_gpu(DIMS, linop_codomain(ElemsLinops[i])->dims, CFL_SIZE);
            debug_printf(DP_INFO, "Runnning ElemsLinops[%d] aaa:\n",i);
            operator_apply(ElemsLinops[i]->forward,N, linop_codomain(ElemsLinops[i])->dims, tmpX2, N, linop_codomain(OSLinop)->dims, tmpX1);
            debug_printf(DP_INFO, "Finished ElemsLinops[%d] aaa\n",i);


            ElemsLinops[i]=linop_chain(OSLinop,ElemsLinops[i]);



            complex float* tmpX = md_alloc_gpu(DIMS, linop_codomain(ElemsLinops[i])->dims, CFL_SIZE);
            debug_printf(DP_INFO, "Runnning ElemsLinops[%d]:\n",i);
            operator_apply(ElemsLinops[i]->forward,N, linop_codomain(ElemsLinops[i])->dims, tmpX, N, mapsDims[i], Elems[i]);
            debug_printf(DP_INFO, "Finished ElemsLinops[%d]\n",i);
            // return 0;
        } else {
            debug_printf(DP_INFO, "LDims[%d]:",i);
            debug_print_dims(DP_INFO,DIMS,LDims[i]);
            debug_printf(DP_INFO, "mapsDims[%d]:",i);
            debug_print_dims(DP_INFO,DIMS,mapsDims[i]);

            md_merge_dims(DIMS, MergedDims, mapsDims[i], LDims[i]);
            md_select_dims(DIMS, ~SquashFlags, NewDims, MergedDims);
            
            debug_printf(DP_INFO, "MergedDims:");
            debug_print_dims(DP_INFO,DIMS,MergedDims);

            long CurFlags=md_nontriv_dims(DIMS,mapsDims[i]);
            long NewFlags=md_nontriv_dims(DIMS,NewDims);
            long TFlags=md_nontriv_dims(DIMS,LDims[i]);
            
            if(mapsDims[i][0]==1) {
                debug_printf(DP_INFO, "ElemsLinop %d: Using fmac on CPU!\n",i);
                ElemsLinops[i] = linop_fmacOnCPU_create(DIMS, MergedDims, ~NewFlags, ~CurFlags, ~TFlags, ElemsL[i]);
            } else {
                ElemsLinops[i] = linop_fmac_create(DIMS, MergedDims, ~NewFlags, ~CurFlags, ~TFlags, ElemsL[i]);
            }

            // Translation
            if(Translation && i==0) {
                long FFTFlags=3;
                const struct linop_s* fft=linop_fftc_create(DIMS, linop_codomain(ElemsLinops[i])->dims, FFTFlags);
                ElemsLinops[0]=linop_chain(ElemsLinops[0],fft);
            }

            if(Translation && i==1) {
                ElemsLinops[1]=linop_fmacOnCPU_create(DIMS, MergedDims, ~NewFlags, ~CurFlags, ~TFlags, ElemsL[i]);
            }
        }
    }
    
    debug_printf(DP_INFO,"Created ElemsLinops\n");

    print_cuda_meminfo();


    long XDims[DIMS];
    long XStrs[DIMS];
    md_copy_dims(DIMS, XDims, linop_codomain(ElemsLinops[0])->dims);
    md_calc_strides(DIMS, XStrs, XDims, CFL_SIZE);
    debug_printf(DP_INFO, "XDims:");
    debug_print_dims(DP_INFO,DIMS,XDims);

    const struct operator_s* uOps[MAX_MAPS];
    const struct operator_s* sOps[MAX_MAPS];
    const struct operator_s* duOps[MAX_MAPS];
    const struct operator_s* dsOps[MAX_MAPS];
    bool ApplyReal[MAX_MAPS];
    bool WrappingTrick[MAX_MAPS];

    // read main linop
    if(gpu) {
        ReadScriptFiles_gpu(&argv[4],num_args-4);
    } else {
        ReadScriptFiles(&argv[4],num_args-4);
    }

    long inputDims_dims[DIMS];
    complex float* inputDims = load_cfl(argv[2], DIMS, inputDims_dims);                
    const struct linop_s* Aop = getLinopScriptFromFile(argv[1],inputDims,inputDims_dims[1]);
    
    long CurDims[DIMS];
    md_copy_dims(DIMS, CurDims, linop_domain(Aop)->dims);
    // md_select_dims(DIMS, ~0, CurDims, XDims);
    debug_printf(DP_INFO, "CurDims:");
    debug_print_dims(DP_INFO,DIMS,CurDims);
    
    long dimsAfterF[DIMS];
    md_copy_dims(DIMS, dimsAfterF, CurDims);
    
    debug_printf(DP_INFO, "Read forward script. dimsAfterF:");
    debug_print_dims(DP_INFO,DIMS,dimsAfterF);
    debug_printf(DP_INFO, "OK linop script reading\n");

    debug_printf(DP_INFO, "XDims:");
    debug_print_dims(DP_INFO,DIMS,XDims);
    // ggg Linop script

    debug_printf(DP_DEBUG1, "ok temp stuff\n");

    print_cuda_meminfo();

    PRINT_TO_FNBUFOUT("Others");
    // complex float* Others = create_cfl(FNbuf, N, XDims);
    PRINT_TO_FNBUFOUT("ThisOne");
    // complex float* ThisOne = create_cfl(FNbuf, N, XDims);
    PRINT_TO_FNBUFOUT("r");
    // complex float* r = create_cfl(FNbuf, N, XDims);
    complex float* Others;
    complex float* ThisOne;
    complex float* r;


    complex float* sLuElems[MAX_MAPS];
    for(i=0;i<nElements;i++) {
        if(gpu) {
#ifdef USE_CUDA
            sLuElems[i] = md_alloc_gpu(DIMS, XDims, CFL_SIZE);
#endif
        } else {
            sLuElems[i] = md_alloc(DIMS, XDims, CFL_SIZE);
        }
    }

    if(gpu) {
#ifdef USE_CUDA
        Others = md_alloc_gpu(DIMS, XDims, CFL_SIZE);
        ThisOne = md_alloc_gpu(DIMS, XDims, CFL_SIZE);
        r = md_alloc_gpu(DIMS, XDims, CFL_SIZE);
#endif
    } else {
        Others = md_alloc(DIMS, XDims, CFL_SIZE);
        ThisOne = md_alloc(DIMS, XDims, CFL_SIZE);
        r = md_alloc(DIMS, XDims, CFL_SIZE);
    }

    debug_printf(DP_DEBUG1, "ok alloc\n");

    PRINT_TO_FNBUFOUT("XDimsOut");
    complex float* XDimsOuta = create_cfl(FNbuf, N, XDims);
    PRINT_TO_FNBUFOUT("mDimsOut");
    complex float* mDimsOuta = create_cfl(FNbuf, N, mapsDims[0]);
    complex float* XDimsOut;
    complex float* mDimsOut;
    if(gpu) {
#ifdef USE_CUDA
        XDimsOut=md_gpu_move(DIMS, XDims, XDimsOuta, CFL_SIZE);
        mDimsOut=md_gpu_move(DIMS, mapsDims[0], mDimsOuta, CFL_SIZE);
#endif
    } else {
        XDimsOut=XDimsOuta;
        mDimsOut=mDimsOuta;
    }

    debug_printf(DP_DEBUG1, "ok tmps create cfl\n");

    complex float* AuxMaps[MAX_MAPS][N_AUX_MAPS];
    for(i=0;i<nElements;i++) {
        for(j=0;j<N_AUX_MAPS;j++) {
            AuxMaps[i][j]=NULL;
        }
    }

    complex float* RandMaps[MAX_MAPS][N_RAND_MAPS_FOR_P];
    for(i=0;i<nElements;i++) {
        for(j=0;j<N_RAND_MAPS_FOR_P;j++) {
            RandMaps[i][j]=NULL;
        }
    }


    // Types: M=1 P=2 D/B=3 T=4 C=5
    for(i=0;i<nElements;i++) {
        CurElementType=round(ElementTypes[i]);
        switch(CurElementType) {
            case 1: // "m type"
                uOps[i] = operator_identity_create(N, mapsDims[i]);
                duOps[i] = operator_ones_create(N, mapsDims[i]);
                sOps[i] = operator_identity_create(N, XDims);
                dsOps[i] = operator_ones_create(N, XDims);
                ApplyReal[i]=true;
                WrappingTrick[i]=false;
                break;
            case 2: // "p type"
                debug_printf(DP_INFO,"Element P\n");
                complex float *tmpMap=md_alloc(DIMS, mapsDims[i], CFL_SIZE);
                for(j=0;j<N_RAND_MAPS_FOR_P;j++) {
#ifdef USE_CUDA
                    if(gpu) {
                        RandMaps[i][j]=md_alloc_gpu(DIMS, mapsDims[i], CFL_SIZE);
                        // md_clear(DIMS, mapsDims[i], RandMaps[i][j], CFL_SIZE);
                    } else
#endif
                    {
                        RandMaps[i][j]=md_alloc(DIMS, mapsDims[i], CFL_SIZE);
                    }
                    float rnd=(frand()*2*M_PI-M_PI);
                    md_zfill(DIMS, mapsDims[i], tmpMap, rnd);
                    md_copy(DIMS, mapsDims[i], RandMaps[i][j], tmpMap, CFL_SIZE);
                    // md_zfill(DIMS, mapsDims[i], RandMaps[i][j], rnd);
                    // md_zsadd(N, mapsDims[i], RandMaps[i][j], RandMaps[i][j], rnd);
                }
                md_free(tmpMap);
                uOps[i] = operator_identity_create(N, mapsDims[i]);
                duOps[i] = operator_ones_create(N, mapsDims[i]);
                sOps[i] = operator_zexp_create(N, XDims);
                dsOps[i] = operator_zexp_create(N, XDims);
                ApplyReal[i]=true;
                WrappingTrick[i]=true;
                debug_printf(DP_INFO,"Element P end\n");
                break;
            case 3: // "D/B type"
                uOps[i] = operator_identity_create(N, mapsDims[i]);
                duOps[i] = operator_ones_create(N, mapsDims[i]);
                sOps[i] = operator_zexp_create(N, XDims);
                dsOps[i] = operator_zexp_create(N, XDims);
                ApplyReal[i]=true;
                WrappingTrick[i]=false;
                break;
            case 4: // "T type"
                debug_printf(DP_DEBUG2,"Element T start %ld\n",i);
                if(gpu) {
#ifdef USE_CUDA
                    // debug_printf(DP_DEBUG2,"Element T aa %ld\n",i);
                    AuxMaps[i][0]=md_alloc_gpu(DIMS, mapsDims[i], CFL_SIZE);
                    // debug_printf(DP_DEBUG2,"Element T ab %ld\n",i);
                    AuxMaps[i][1]=md_alloc_gpu(DIMS, mapsDims[i], CFL_SIZE);
                    // debug_printf(DP_DEBUG2,"Element T ac %ld\n",i);
#endif
                } else {
                    AuxMaps[i][0]=md_alloc(DIMS, mapsDims[i], CFL_SIZE);
                    AuxMaps[i][1]=md_alloc(DIMS, mapsDims[i], CFL_SIZE);
                }
                // md_clear(DIMS, mapsDims[i], AuxMaps[i][0], CFL_SIZE);

                md_clear(N, mapsDims[i], AuxMaps[i][0], CFL_SIZE);
                md_zexpj(N, mapsDims[i], AuxMaps[i][0], AuxMaps[i][0]);

                md_zexpj(N, mapsDims[i], AuxMaps[i][1], AuxMaps[i][0]);
                md_zsmul(N, mapsDims[i], AuxMaps[i][1], AuxMaps[i][1], -1);

                // debug_printf(DP_DEBUG2,"Element T ad %ld\n",i);
                // md_zfill(DIMS, mapsDims[i], AuxMaps[i][0], One);
                // debug_printf(DP_DEBUG2,"Element T ae %ld\n",i);
                // md_zfill(DIMS, mapsDims[i], AuxMaps[i][1], -One);
                // debug_printf(DP_DEBUG2,"Element T a %ld\n",i);

                // uOps[i] = operator_spow_create(N, mapsDims[i],-1);
                uOps[i] = operator_zrdiv_create(N, mapsDims[i],AuxMaps[i][0]);
                // const struct operator_s* InvSqrOp = operator_spow_create(N, mapsDims[i],-2);
                const struct operator_s* SqrOp = operator_zsqr_create(N, mapsDims[i]);
                // const struct operator_s* MinusOp = operator_smul_create(N, mapsDims[i],-1);
                const struct operator_s* MinusInv = operator_zrdiv_create(N, mapsDims[i],AuxMaps[i][1]);
                // duOps[i] = operator_chain(InvSqrOp,MinusOp);
                // const struct operator_s* InvSqrOp; =operator_chain(SqrOp,uOps[i]);
                duOps[i] = operator_chain(SqrOp,MinusInv);  
                sOps[i] = operator_zexp_create(N, XDims);
                dsOps[i] = operator_zexp_create(N, XDims);
                ApplyReal[i]=true;
                WrappingTrick[i]=false;
                debug_printf(DP_DEBUG2,"Element T end %ld\n",i);
                break;
            case 5: // "C type" - just like m except without Re()
                uOps[i] = operator_identity_create(N, mapsDims[i]);
                duOps[i] = operator_ones_create(N, mapsDims[i]);
                sOps[i] = operator_identity_create(N, XDims);
                dsOps[i] = operator_ones_create(N, XDims);
                ApplyReal[i]=false;
                WrappingTrick[i]=false;
                break;
            case 6: // "m type, no abs"
                uOps[i] = operator_identity_create(N, mapsDims[i]);
                duOps[i] = operator_ones_create(N, mapsDims[i]);
                sOps[i] = operator_identity_create(N, XDims);
                dsOps[i] = operator_ones_create(N, XDims);
                ApplyReal[i]=true;
                WrappingTrick[i]=false;
                break;
            default : 
                debug_printf(DP_INFO,"Wrong element type %ld\n",CurElementType);
                return 0;
                break;
        }
        // debug_printf(DP_INFO,"Running uOp %ld\n",i);
        // operator_apply(uOps[i],N, mapsDims[i], tmpElems[i], N, mapsDims[i], Elems[i]);
    }
    

    const struct operator_s* XConjOp = operator_zconj_create(N, XDims);
    

    debug_printf(DP_INFO,"Created s,ds ops\n");

    print_cuda_meminfo();

    const struct operator_s* sLuOps[MAX_MAPS];
    const struct operator_s* dsLuOps[MAX_MAPS];
    for(i=0;i<nElements;i++) {
        debug_printf(DP_DEBUG2,"Creating sLu, dsLu %d\n",i);
        debug_printf(DP_DEBUG2,"u[%d] domain  :",i);
        debug_print_dims(DP_DEBUG2,DIMS,operator_codomain(uOps[i])->dims);
        debug_printf(DP_DEBUG2,"L[%d] domain  :",i);
        debug_print_dims(DP_DEBUG2,DIMS,operator_domain(ElemsLinops[i]->forward)->dims);
        debug_printf(DP_DEBUG2,"L[%d] codomain:",i);
        debug_print_dims(DP_DEBUG2,DIMS,operator_codomain(ElemsLinops[i]->forward)->dims);
        debug_printf(DP_DEBUG2,"s[%d] domain:  ",i);
        debug_print_dims(DP_DEBUG2,DIMS,operator_domain(sOps[i])->dims);
        sLuOps[i]=operator_chain(operator_chain(uOps[i],ElemsLinops[i]->forward),sOps[i]);
        // sLuOps[i]=ElemsLinops[i]->forward;
        dsLuOps[i]=operator_chain(operator_chain(operator_chain(uOps[i],ElemsLinops[i]->forward),dsOps[i]),XConjOp);
    }
    debug_printf(DP_DEBUG2,"ok s,ds\n");

    // Build proximals
    const struct operator_p_s* prox_ops[MAX_MAPS];
    struct iter_op_p_s it_prox_ops[MAX_MAPS];
    // debug_printf(DP_INFO, "l1-wavelet regularization: %f randshift %ld\n", regs[nr].lambda,randshift);

    const long sMINSIZE=8;
    // const long sMINSIZE=4;

    long minsize[DIMS] = { [0 ... DIMS - 1] = 1 };
    // minsize[0] = MIN(mapsDims[0][0], sMINSIZE);
    // minsize[1] = MIN(mapsDims[0][1], sMINSIZE);
    //minsize[2] = MIN(mapsDims[0][2], 16);

    unsigned int wflags = 0;
    for (j = 0; j < DIMS; j++) {
        if ((1 < mapsDims[0][j]) && MD_IS_SET(xflags, j)) {
            wflags = MD_SET(wflags, j);
            minsize[j] = MIN(mapsDims[0][j], sMINSIZE);
        }
    }

    struct linop_s* ProxLinops[MAX_MAPS];
    complex float* AuxProxMaps[MAX_MAPS];

    for(i=0;i<nElements;i++) {
        ProxLinops[i]=NULL;
        AuxProxMaps[i]=NULL;
        long CurProxType=round(ProxTypes[i]);

        float CurLambda=creal(ElemsLambda[i]);

        long blkdims[MAX_LEV][DIMS];
        int levels;
        int remove_mean = 0;
        bool overlapping_blocks = false;

        switch (CurProxType) {
            case 0:
                prox_ops[i] = prox_wavelet_thresh_create(DIMS, mapsDims[i], wflags, jflags, minsize,CurLambda , randshift); 
                break;
            case 1:
                prox_ops[i] = prox_thresh_create(DIMS, mapsDims[i], CurLambda, xflags);
                break;
            case 2:
                /*debug_printf(DP_INFO,"L2 a\n");
                ProxLinops[i]=linop_gradm_create(DIMS, mapsDims[i], xflags);
                debug_printf(DP_INFO,"L2 b\n");
                prox_ops[i]=prox_leastsquares_create(DIMS, linop_codomain(ProxLinops[i])->dims, CurLambda, NULL);
                debug_printf(DP_INFO,"L2 c\n");
                prox_ops[i]=operator_chain(ProxLinops[i]->forward,prox_ops[i]);
                debug_printf(DP_INFO,"L2 d\n");
                prox_ops[i]=operator_chain(prox_ops[i],ProxLinops[i]->adjoint);*/

                prox_ops[i] = prox_leastsquares_create(DIMS, mapsDims[i], CurLambda, NULL);
                break;
            case 3: 
                prox_ops[i] = prox_l2norm_create(DIMS, mapsDims[i], CurLambda);
                break;
            case 4:                
                levels = llr_blkdims(blkdims, Ljflags, mapsDims[i], llr_blk);

                prox_ops[i] = lrthresh_create(mapsDims[i], randshift, wflags, (const long (*)[DIMS])blkdims, CurLambda, false, remove_mean, overlapping_blocks);
                break;
            default:
                error("Error prox type for element %d", i);
                break;
        }

        it_prox_ops[i]=OPERATOR_P2ITOP(prox_ops[i]);
    }

    long ErrOutDims[DIMS];
    for(i=0;i<DIMS;i++) {
        ErrOutDims[i]=1; }

    ErrOutDims[0]=ERR_VEC_LENGTH;
    PRINT_TO_FNBUFOUT("ErrVec");
    complex float* ErrOut = create_cfl(FNbuf, N, ErrOutDims);
    md_clear(N, ErrOutDims, ErrOut, CFL_SIZE);
    //complex float* ErrOut = create_cfl("/tmp/ErrVec", N, ErrOutDims);

    debug_printf(DP_DEBUG1, "ok copy warmstarts\n");

    float h=1;
    long k=0;
    long K=10;

    long iter;
    long inneriter;
    long curElem;

    long ninnerIterToRun;

    float curerr;
    float rmsMap;
    long ContIter=0;

    // md_zfill(N, XDims, XDimsOuta, One);
    debug_printf(DP_DEBUG1, "Initializing others\n");
    print_cuda_meminfo();
    // md_copy(DIMS, XDims, Others, XDimsOuta, CFL_SIZE);
    // md_clear(N, XDims, Others, CFL_SIZE);
    // md_zexpj(N, XDims, Others, Others);
    
    debug_printf(DP_DEBUG2, "Maps dims:\n");
    for(j=0;j<nElements;j++) {
        debug_print_dims(DP_DEBUG2,DIMS,mapsDims[j]);
    }

    for(j=0;j<nElements;j++) {
        debug_printf(DP_DEBUG1, "Calc sLu [%d]\n",j);
        
        // debug_printf(DP_DEBUG1, "Calc u [%d]\n",j);
        // operator_apply(uOps[j],N, mapsDims[j], pElems[j], N, mapsDims[j], Elems[j]);
        // debug_printf(DP_DEBUG1, "Calc L [%d]\n",j);
        // operator_apply(ElemsLinops[j]->forward,N, XDims, Others, N, mapsDims[j], pElems[j]);
        // debug_printf(DP_DEBUG1, "Calc s [%d]\n",j);
        // operator_apply(sOps[j],N, XDims, ThisOne, N, XDims, Others);

        // operator_apply(sLuOps[j],N, XDims, ThisOne, N, mapsDims[j], Elems[j]);
        operator_apply(sLuOps[j],N, XDims, sLuElems[j], N, mapsDims[j], Elems[j]);
        debug_printf(DP_DEBUG1, "Mul to Others\n");
        // md_zmul(DIMS, XDims, Others, Others, ThisOne);
        // md_zmul(DIMS, XDims, Others, Others, sLuElems[j]);
        // md_zmul(DIMS, XDims, Others, Others, XDimsOuta);
    }
    print_cuda_meminfo();

    // Translation
    long XFFTFlags=3;
    const struct linop_s* XF2DOp = linop_fftc_create(DIMS, XDims, XFFTFlags);

    debug_printf(DP_DEBUG1, "Calculating 'Others'\n");
    CalcOthers(N, XDims, Others, nElements, -1, sLuElems,Translation,XF2DOp->forward);
    print_cuda_meminfo();
    
    debug_printf(DP_DEBUG1, "Apply first normal\n");
    operator_apply(Aop->normal,N, XDims, r, N, XDims, Others);

    float rms_sig_adj=md_zrms(N, XDims, sig_adj);
    float rms_Est0_adj=md_zrms(N, XDims, r);

    debug_printf(DP_INFO, "RMS: sig_adj: %g, Est0 %g\n",rms_sig_adj,rms_Est0_adj);
    print_cuda_meminfo();
    //if(rms_Est0_adj>0) {
    //  float RMSFactor=rms_sig_adj/rms_Est0_adj;
    //  debug_printf(DP_INFO, "Correcting with factor %g\n",RMSFactor);
    //  md_zsmul(DIMS, mapsDims[0], Elems[0], Elems[0], RMSFactor);

    //  md_copy(DIMS, XDims, Others, XDimsOuta, CFL_SIZE);
    //  debug_printf(DP_DEBUG1, "Calculating 'Others'\n");

    //  for(j=0;j<nElements;j++) {
    //      debug_printf(DP_DEBUG1, "Calc sLu\n");
    //      operator_apply(sLuOps[j],N, XDims, ThisOne, N, mapsDims[j], Elems[j]);
    //      debug_printf(DP_DEBUG1, "Mul to Others\n");
    //      md_zmul(DIMS, XDims, Others, Others, ThisOne);
    //      // md_zmul(DIMS, XDims, Others, Others, XDimsOuta);
    //  }
    //  debug_printf(DP_DEBUG1, "Apply first normal\n");
    //  operator_apply(Aop->normal,N, XDims, r, N, XDims, Others);
    //}

    debug_printf(DP_DEBUG1, "r=AHsig-r\n");
    md_zsub(N, XDims, r, sig_adj, r);

    curerr=md_zrms(N, XDims, r);
    debug_printf(DP_INFO, "Initial rmse %g\n",curerr);

    // zsum timing test
    // debug_printf(DP_INFO, "zsum timing test\n");
    // // long XXDims[DIMS];
    // // complex float* XX;
    // // PRINT_TO_FNBUF("XX");
    // // XX = load_cfl(FNbuf, N, XXDims); 
    // float sum=0;
    // for(i=0;i<20;i++) {
    //     debug_printf(DP_INFO, "zsum timing test %d %g\n",i,sum);
    //     sum=md_asum(N,XDims,r);
    // }
    // debug_printf(DP_INFO, "zsum scalar mult test\n");
    // for(i=0;i<20;i++) {
    //     debug_printf(DP_INFO, "sum mul test %d\n",i);
    //     md_smul(N, XDims, r, r, 2.0f);
    // }
    // for(i=0;i<20;i++) {
    //     debug_printf(DP_INFO, "zsum mul test %d\n",i);
    //     md_zsmul(N, XDims, r, r, 2.0f);
    // }
    // for(i=0;i<20;i++) {
    //     debug_printf(DP_INFO, "md_zmul test %d\n",i);
    //     md_zmul(N, XDims, r, ThisOne,ThisOne);
    // }

    // long OStrs[N];
    // long IStrs[N];
    // long TStrs[N];
    // md_calc_strides(N, IStrs, mapsDims[1], CFL_SIZE);
    // md_calc_strides(N, OStrs, XDims, CFL_SIZE);
    // md_calc_strides(N, TStrs, LDims[1], CFL_SIZE);

    // for(i=0;i<20;i++) {
    //     debug_printf(DP_INFO, "md_zfmac test %d\n",i);
    //     md_zfmac2(N, XDims, OStrs, r, IStrs, Elems[1], TStrs, ElemsL[1]);
    //     // operator_apply(ElemsLinops[1]->forward,N, XDims, r, N, mapsDims[1], Elems[1]);
    // }
    

    // for(i=0;i<20;i++) {
    //     debug_printf(DP_INFO, "Linop f test %d\n",i);
    //     operator_apply(ElemsLinops[1]->forward,N, XDims, r, N, mapsDims[1], Elems[1]);
    // }
    // for(i=0;i<20;i++) {
    //     debug_printf(DP_INFO, "Linop a test %d\n",i);
    //     operator_apply(ElemsLinops[1]->adjoint,N, mapsDims[1], Elems[1], N, XDims, r);
    // }
    //     // md_zfmac();
    //     // md_clear2(data->N, data->odims, data->ostrs, dst, CFL_SIZE);
    //     // md_zfmac2(data->N, data->dims, data->ostrs, dst, data->istrs, src, data->tstrs, tensor);

    //     // debug_printf(DP_INFO, "zfmac mul test %d\n",i);

    //     // ElemsLinops[i]

    //     // md_zsmul(N, XDims, r, r, 2.0f);
    // // }

    // debug_printf(DP_INFO, "zsum timing end\n");

    float Xrms,Xrms2,RelMapChangeWithAlphaAfterProx,Xrms4;

    complex float CurAlpha;

    double Start_time = timestamp();
    double last_save_time = timestamp();

    long m;
    double cur_time,cur_time1,cur_time2;
    double end_time;

    // md_copy(DIMS, mapsDims[1], mDimsOut, Elems[1], CFL_SIZE); // ok

    double TotalTime_Normal=0.0;
    double TotalTime_a=0.0;
    double TotalTime_b=0.0;
    double TotalTime_c=0.0;
    double TotalTime_d=0.0;

    double Before_a,Before_b,Before_c,Before_d;

    bool DoNotSave=false;
    bool StopIterations=false;

    bool Calc_r_this_time;

    long SumInnerIter=0;
    for(j=0;j<nElements;j++) { SumInnerIter+=creal(ninneriter[j]); }
    
    long NextLineSearch[MAX_MAPS];
    for(j=0;j<nElements;j++) { NextLineSearch[j]=0; }
    float FinalBetaFactor=0.01f;
    float StartRMSRatio=0.1f;
    float BetaDecay=0.5f;
    float MinEstChange=1e-6;
    long MaxLineSearchIters=50;
    long MaxItersWithoutLineSearch=300;
    float MaxEstChangeToNextLineSearch=0.1f;
    
    float CurRegCosts[MAX_MAPS];
    /*long XFlags=md_nontriv_dims(DIMS,mapsDims[0]);
    struct linop_s* LL = linop_fmac_create(DIMS, mapsDims[0], ~XFlags, ~XFlags, ~XFlags, Elems[1]);
    operator_apply(LL->forward,N, mapsDims[0], pElems[0], N, mapsDims[0], Elems[0]);
    md_copy(DIMS, mapsDims[0], Elemsa[0], pElems[0], CFL_SIZE);
    for(i=0;i<4;i++) {
        debug_printf(DP_INFO,"XA[%d]=%f\n",i,crealf(Elemsa[0][i]));
    }
    set_fmac_tensor(LL, Elems[2]);
    operator_apply(LL->forward,N, mapsDims[0], pElems[0], N, mapsDims[0], Elems[0]);
    md_copy(DIMS, mapsDims[0], Elemsa[0], pElems[0], CFL_SIZE);
    for(i=0;i<4;i++) {
        debug_printf(DP_INFO,"XB[%d]=%f\n",i,crealf(Elemsa[0][i]));
    }*/

    print_cuda_meminfo();

    struct operator_s* ExtraTrasOp=NULL;
    struct operator_s* ExtraTrasOpAdj=NULL;

    if(Translation) {
        ExtraTrasOp=XF2DOp->adjoint;
        ExtraTrasOpAdj=XF2DOp->forward;
    }

    long XFlags=md_nontriv_dims(DIMS,XDims);
    struct linop_s* OthersAnddsLu = linop_fmac_create(DIMS, XDims, ~XFlags, ~XFlags, ~XFlags, ThisOne);
    const struct linop_s* FixedDu[MAX_MAPS];
    struct operator_s* OpForLipschitz[MAX_MAPS];
    long mapFlags[MAX_MAPS];
    for(i=0;i<nElements;i++) {
        mapFlags[i]=md_nontriv_dims(DIMS,mapsDims[i]);
        FixedDu[i] = linop_fmac_create(DIMS, mapsDims[i], ~mapFlags[i], ~mapFlags[i], ~mapFlags[i], Elems[i]);
        
        if(Translation) {
            OpForLipschitz[i] = operator_chain(operator_chain(operator_chain(operator_chain(operator_chain(operator_chain(operator_chain(operator_chain(
                FixedDu[i]->forward,
                ElemsLinops[i]->forward),
                OthersAnddsLu->forward),
                ExtraTrasOp), // Translation
                Aop->normal),
                ExtraTrasOpAdj), // Translation
                OthersAnddsLu->adjoint),
                ElemsLinops[i]->adjoint),
                FixedDu[i]->adjoint);
        } else {
            OpForLipschitz[i] = operator_chain(operator_chain(operator_chain(operator_chain(operator_chain(operator_chain(
                FixedDu[i]->forward,
                ElemsLinops[i]->forward),
                OthersAnddsLu->forward),
                Aop->normal),
                OthersAnddsLu->adjoint),
                ElemsLinops[i]->adjoint),
                FixedDu[i]->adjoint);
        }
    }

    

    // Power method
    long PowerMethod_nIters;
    // md_clear(N, XDims, Others, CFL_SIZE);
    // float AopNMaxEig=GetMaxEigPowerMethod(Aop->normal, N, XDims,ThisOne, Others,10, 1e-2,&PowerMethod_nIters,1.0f);
    // debug_printf(DP_INFO,"Aop->normal MaxEig %f in %d iters\n",AopNMaxEig,PowerMethod_nIters);
    // return 0;
    
    print_cuda_meminfo();

#ifdef USE_CUDA
    if(cuda_ondevice(Others)) { debug_printf(DP_DEBUG1,"Others on GPU\n");} else {  debug_printf(DP_DEBUG1,"Others on CPU\n");  }
    if(cuda_ondevice(ThisOne)) {    debug_printf(DP_DEBUG1,"ThisOne on GPU\n");} else { debug_printf(DP_DEBUG1,"ThisOne on CPU\n"); }
    if(cuda_ondevice(r)) {  debug_printf(DP_DEBUG1,"r on GPU\n");} else {   debug_printf(DP_DEBUG1,"r on CPU\n");   }
    if(cuda_ondevice(Elems[0])) {   debug_printf(DP_DEBUG1,"Elems[0] on GPU\n");} else {    debug_printf(DP_DEBUG1,"Elems[0] on CPU\n");    }
    if(cuda_ondevice(pElems[0])) {  debug_printf(DP_DEBUG1,"pElems[0] on GPU\n");} else {   debug_printf(DP_DEBUG1,"pElems[0] on CPU\n");   }
    if(cuda_ondevice(tmpElems[0])) {    debug_printf(DP_DEBUG1,"tmpElems[0] on GPU\n");} else { debug_printf(DP_DEBUG1,"tmpElems[0] on CPU\n"); }
    if(cuda_ondevice(duElems[0])) { debug_printf(DP_DEBUG1,"duElems[0] on GPU\n");} else {  debug_printf(DP_DEBUG1,"duElems[0] on CPU\n");  }
#endif
    double IterStartTime = timestamp();
    for(iter=0;iter<maxiter;iter++) {
        Before_a = timestamp();
        for(curElem=0;curElem<nElements;curElem++) {
            CurElementType=round(ElementTypes[curElem]);
            ninnerIterToRun=creal(ninneriter[curElem]);
            if(ninnerIterToRun>0) 
            {
                debug_printf(DP_DEBUG3, "Initializing 'Others' %d\n",curElem);

                // if(Calc_r_every_time) {
                if(iter<Calc_r_once_per_iter_StartIter)  {
                    Calc_r_this_time=true;
                } else {
                    Calc_r_this_time= (curElem==0);
                }
                
                CalcOthers(N, XDims, Others, nElements, curElem, sLuElems,Translation,XF2DOp->forward);
                // if(Calc_r_this_time)  {
                // Here calculating "Other" - do anyway
                // {
                //     md_clear(N, XDims, Others, CFL_SIZE);
                //     md_zexpj(N, XDims, Others, Others);

                //     debug_printf(DP_DEBUG3, "Calculating 'Others' %d\n",curElem);

                //     // cur_time1=  timestamp();
                //     for(j=0;j<nElements;j++) {
                //         if(j!=curElem) {
                //             md_zmul(DIMS, XDims, Others, Others, sLuElems[j]);
                //         }
                //     }
                // }
                
                debug_printf(DP_DEBUG3, "Innershots %d\n",curElem);
            }

            for(inneriter=0;inneriter<ninnerIterToRun;inneriter++) {
                Before_c = timestamp();

                operator_apply(sLuOps[curElem],N, XDims, sLuElems[curElem], N, mapsDims[curElem], Elems[curElem]);

                if(Calc_r_this_time)  {
                    debug_printf(DP_DEBUG3, "add this one\n");

                    md_zmul(DIMS, XDims, r, sLuElems[curElem], Others);

                    if(hasGT) {
                        if(inneriter==0) {
                            debug_printf(DP_DEBUG3, "Compare to GT\n");
                            md_zsub(N, XDims, XDimsOut, r, GT);
                            Xrms=md_zrms(N, XDims, XDimsOut)/GTrms;
                        }
                    }

                    // Translation
                    if(Translation) {
                        operator_apply(XF2DOp->adjoint,N, XDims, r, N, XDims, r); }

                    double BeforeNormal = timestamp();
                    debug_printf(DP_DEBUG3, "Apply normal\n");
                    operator_apply(Aop->normal,N, XDims, r, N, XDims, r);
                    TotalTime_Normal += (timestamp()-BeforeNormal);
                    
                    // Xrms=md_zrms(N, XDims, r);
                    // debug_printf(DP_INFO, "AHACurEst rms %g\n",Xrms);

                    debug_printf(DP_DEBUG3, "r=AHsig-r\n");
                    md_zsub(N, XDims, r, sig_adj, r);
                    // Now we have r

                    TotalTime_c += (timestamp()-Before_c);
                    Before_b = timestamp();
                    // d here - 140-150
                    Before_d = timestamp();

                    curerr=md_zrms(N, XDims, r);
                    //curerr=0.00000001;

                    ErrOut[ContIter++]=curerr;

                    if(ContIter>iter_to_tol) {
                        if(!  (curerr< creal(ErrOut[ContIter-iter_to_tol])*tol )  )  {  // supposes to take care of nan too
                            debug_printf(DP_WARN, "Err bigger than tolerance. Stopping.\n");
                            StopIterations=true;
                            if(! (curerr>=0) ) {
                                DoNotSave=true;
                                debug_printf(DP_WARN, "!(err>0) !!! Not saving maps\n");
                            }
                        }
                    }

                    // Translation
                    if(Translation) {
                        operator_apply(XF2DOp->forward,N, XDims, r, N, XDims, r); }
                        
                    // Apply main linop normal, get r
                    // AHACurEst= bart(BARTS_Aop.cmd,BARTS_Aop.ImSz16,CurEst,BARTS_Aop.Others{:});
                    //    r=ksp_adj- AHACurEst;
                } // if(Calc_r_this_time)  {


                // t = Pt(t + alphat * real(-  (T' * (conj(Mm) .* conj(expTt) .* r)) .* ((1./t).^2)    ), alphat);
                debug_printf(DP_DEBUG3, "Applying dsLuOp %ld s\n",curElem);
                switch(CurElementType) {
                    case 1: // "m type"
                    case 5: // "C type" - just like m except without Re()
                        md_clear(N, XDims, ThisOne, CFL_SIZE);
                        md_zexp(N, XDims, ThisOne, ThisOne);
                        break;
                    case 2:
                    case 3:
                    case 4:
                        md_zconj(N, XDims, ThisOne, sLuElems[curElem]);
                        break;
                    default:
                        operator_apply(dsLuOps[curElem],N, XDims, ThisOne, N, mapsDims[curElem], Elems[curElem]);
                        break;
                }
                    
                // if(Calc_r_every_time) {
                complex float * rOthers;
                if(iter<Calc_r_once_per_iter_StartIter) {
                    rOthers=r;
                    // md_zmulc(DIMS, XDims, r, r, Others);
                } else {
                    rOthers=Others;
                    // md_zmulc(DIMS, XDims, Others, r, Others);
                }
                md_zmulc(DIMS, XDims, rOthers, r, Others);

                debug_printf(DP_DEBUG3, "Applying adj Linop\n");
                // if(Calc_r_every_time) {
                // if(iter<Calc_r_once_per_iter_StartIter) {
                //     // md_zmul(DIMS, XDims, r, r, ThisOne);
                //     operator_apply(ElemsLinops[curElem]->adjoint,N,mapsDims[curElem],tmpElems[curElem],N,XDims,r);
                // } else {
                //     // md_zmul(DIMS, XDims, Others, Others, ThisOne);
                //     operator_apply(ElemsLinops[curElem]->adjoint,N,mapsDims[curElem],tmpElems[curElem],N,XDims,Others);
                // }
                md_zmul(DIMS, XDims, rOthers, rOthers, ThisOne);
                // if(Translation) {
                //     if(curElem<nElements-1) {
                //         operator_apply(XF2DOp->adjoint,N,XDims,rOthers,N,XDims,rOthers);
                //     }
                // }
                operator_apply(ElemsLinops[curElem]->adjoint,N,mapsDims[curElem],tmpElems[curElem],N,XDims,rOthers);

                debug_printf(DP_DEBUG3, "Applying duOp\n");
                operator_apply(duOps[curElem],N,mapsDims[curElem],duElems[curElem],N,mapsDims[curElem],Elems[curElem]);

                // // Lipshitz here
                // if(UseLipshitz) {
                // // ThisOne contains (conj??) dsLuOps, duElems contain current du
                //     // !!! might need to switch zmul/zmulc regarding ThisOne
                // // Other available in calcEveryTime, otherwise recalculate it
                // // We need X=Others*dsLuOp
                // // get R=rand in size of cur map dims
                // // Then R*du, apply Li on that, multiply by X, apply normal on that, then multiply by conj(X), LiH, *du
                // // pElems are available
                // // pElems will be the randome vector through the power iterations
                // if((iter%100)==99) {
                //     debug_printf(DP_DEBUG3, "Lipshitz start %d\n",curElem);
                //     long MaxLIters=20;

                //     if(!Calc_r_every_time) {
                //         md_clear(N, XDims, Others, CFL_SIZE);
                //         md_zexpj(N, XDims, Others, Others);
                //         for(j=0;j<nElements;j++) {
                //             if(j!=curElem) {    md_zmul(DIMS, XDims, Others, Others, sLuElems[j]); } }
                //     }

                //     // ADD md_zmul(N, ThisOne, Others, ThisOne)
                //     // OR 
                //     md_zmulc(N, ThisOne, Others, ThisOne);
                //     set_fmac_tensor(OthersAnddsLu, ThisOne);
                //     set_fmac_tensor(FixedDu[curElem], duElems[curElem]);
                //     float MaxEig=GetMaxEigPowerMethod(OpForLipschitz[curElem], N, mapsDims[curElem],tmpElems2[curElem], pElems[curElem],50, 1e-2);

                //     md_copy(DIMS, XDims, XDimsOut, r, CFL_SIZE); // ok
                //     md_copy(DIMS, mapsDims[curElem], Elems0[curElem], tmpElems[curElem], CFL_SIZE); // ok

                //     debug_printf(DP_DEBUG3, "Creating rand\n");
                //     for (i = 0; i < md_calc_size(N, mapsDims[curElem]); i++) {
                //         Elemsa[curElem][i] = uniform_rand();
                //     }
                //     // md_gaussian_rand(1, MD_DIMS(data->size_x / 2), x);
                //     md_copy(DIMS, mapsDims[curElem], pElems[curElem], Elemsa[curElem], CFL_SIZE); // ok

                //     float L,ErrRMSRatio,CurRMS,CurRMSAx,nom_ir,denom_ir,NewAlpha;
                //     complex float denom;
                //     complex float nom;
                    
                //     CurRMS=md_zrms(N, mapsDims[curElem], pElems[curElem]);
                //     debug_printf(DP_DEBUG3, "RandRMS=%e\n",CurRMS);

                //     for (i = 0; i < MaxLIters; i++) {
                //         // debug_printf(DP_INFO, "Power iter %d start\n",i);
                //         // one iter
                //         // md_zmul(N, mapsDims[curElem], tmpElems[curElem], duElems[curElem], pElems[curElem]);
                //         md_copy(DIMS, mapsDims[curElem], tmpElems[curElem], pElems[curElem], CFL_SIZE);
                //         operator_apply(ElemsLinops[curElem]->forward,N,XDims,r,N,mapsDims[curElem],tmpElems[curElem]);
                //         REMOVE md_zmul(N, XDims, r, r, Others);
                //         md_zmul(N, XDims, r, r, ThisOne);
                //         operator_apply(Aop->normal,N, XDims, r, N, XDims, r);
                //         REMOVE md_zmulc(N, XDims, r, r, Others);
                //         md_zmulc(N, XDims, r, r, ThisOne);
                //         operator_apply(ElemsLinops[curElem]->adjoint,N,mapsDims[curElem],tmpElems[curElem],N,XDims,r);
                //         // md_zmul(N, mapsDims[curElem], tmpElems[curElem], duElems[curElem], tmpElems[curElem]);
                //         // now check how the factor and err rms
                //         // [a/b a*conj(b.')/(norm(b).^2) a*conj(b.')/sum(b*conj(b.')) ]
                //         denom=md_zscalar(N, mapsDims[curElem], pElems[curElem], pElems[curElem]);
                //         nom=md_zscalar(N, mapsDims[curElem], tmpElems[curElem], pElems[curElem]);
                        
                //         nom_ir=cimagf(nom)/crealf(nom);
                //         denom_ir=cimagf(denom)/crealf(denom);
                //         L=crealf(nom)/crealf(denom);
                        
                //         md_zsmul(N, mapsDims[curElem], pElems[curElem], pElems[curElem], L);
                //         md_zsub(N, mapsDims[curElem], pElems[curElem], tmpElems[curElem], pElems[curElem]); // optr = iptr1 - iptr2
                        
                //         CurRMS=md_zrms(N, mapsDims[curElem], pElems[curElem]);
                //         CurRMSAx=md_zrms(N, mapsDims[curElem], tmpElems[curElem]);
                //         if(nom_ir>1e-5 || denom_ir>1e-5) { debug_printf(DP_INFO, "Nom,Denom i/r %6.2e %6.2e\n",nom_ir,denom_ir); }
                //         debug_printf(DP_DEBUG3, "CurRMS=%6.2e, CurRMSAx=%6.2e, nom=%6.2e denom=%6.2e\n",CurRMS,CurRMSAx,crealf(nom),crealf(denom));
                //         ErrRMSRatio=CurRMS/CurRMSAx;

                //         if(ErrRMSRatio<1e-2) break;

                //         md_copy(DIMS, mapsDims[curElem], pElems[curElem], tmpElems[curElem], CFL_SIZE); // ok
                //         md_zsmul(N, mapsDims[curElem], pElems[curElem], pElems[curElem], 1.0f/CurRMSAx);
                //         // end one iter
                //         debug_printf(DP_DEBUG3, "Power iter %3d L=%6.2e Rerr=%6.2e\n",i,L,ErrRMSRatio);
                //     }

                //     NewAlpha=sqrtf(1.0f/L);
                //     debug_printf(DP_INFO, "Lipschitz: Elem %d Piter %d NewA %6.2e      L=%6.2e Rerr=%6.2e\n",curElem,i,NewAlpha,L,ErrRMSRatio);
                //     ElemsAlpha[curElem]=NewAlpha;

                //     // return saved copy of r to continue as usual 
                //     md_copy(DIMS, XDims, r, XDimsOut, CFL_SIZE); // ok
                //     md_copy(DIMS, mapsDims[curElem], tmpElems[curElem], Elems0[curElem], CFL_SIZE); // ok
                //     debug_printf(DP_DEBUG3, "Lipshitz end\n");
                //     // StopIterations=true;
                // }
                // }
                // // End Lipshitz here

                debug_printf(DP_DEBUG4, "Applying multiplications\n");
                md_zmul(N, mapsDims[curElem], tmpElems[curElem], tmpElems[curElem], duElems[curElem]);

                // md_copy(DIMS, mapsDims[curElem], mDimsOut, tmpElems[curElem], CFL_SIZE); // ok
                // dd 2

                debug_printf(DP_DEBUG4, "Applying real\n");
                if(ApplyReal[curElem]) {
                    md_zreal(N, mapsDims[curElem], tmpElems[curElem], tmpElems[curElem]);
                } else {
                    // nothing
                }
                // md_zreal(N, mapsDims[curElem], tmpElems[curElem], tmpElems[curElem]);
                
                debug_printf(DP_DEBUG3, "Applying rms\n");
                Xrms2=md_zrms(N, mapsDims[curElem], tmpElems[curElem])/md_zrms(N, mapsDims[curElem], Elems[curElem]);
                
                // Line search
                if(UseLineSearch || UseLipshitz) {
                    if(iter==NextLineSearch[curElem] && inneriter==0) {
                        float CurTryRMS,LastTryRMS,CurBeta,EstChange,NewAlpha,AlphaRatio;
                        float MaxEig;

                        long ItersToNextLineSearch;

                        CurBeta=StartRMSRatio/Xrms2;

                        // debug_printf(DP_INFO, "Line search %d\n",curElem);
                        // if(!Calc_r_every_time) {
                        if(iter>=Calc_r_once_per_iter_StartIter) {
                            CalcOthers(N, XDims, Others, nElements, curElem, sLuElems,Translation,XF2DOp->forward);
                            // md_clear(N, XDims, Others, CFL_SIZE);
                            // md_zexpj(N, XDims, Others, Others);
                            // for(j=0;j<nElements;j++) {
                            //     if(j!=curElem) {    md_zmul(DIMS, XDims, Others, Others, sLuElems[j]); } }
                        }
                        // Now we have others
                        // ThisOne contains (conj??) dsLuOps, duElems contain current du
                        // ThisOne, pElems, duElems is available
                        if(UseLipshitz) {
                            // debug_printf(DP_INFO, "Calling power method for Lipschitz\n");
                            md_zmulc(N, XDims, ThisOne, Others, ThisOne);
                            set_fmac_tensor(OthersAnddsLu, ThisOne);
                            set_fmac_tensor(FixedDu[curElem], duElems[curElem]);
                            // debug_printf(DP_DEBUG4,"AAA %e %e %e",md_zrms(N, XDims, ThisOne),md_zrms(N, mapsDims[curElem], duElems[curElem]),5.0f);
                            // MaxEig=GetMaxEigPowerMethod(OpForLipschitz[curElem], N, mapsDims[curElem],tmpElems2[curElem], pElems[curElem],50, 1e-2,&PowerMethod_nIters);
                            float SlackFactor=1e-2;
                            MaxEig=GetMaxEigPowerMethod(OpForLipschitz[curElem], N, mapsDims[curElem],tmpElems2[curElem], tmpMapForPowerMethod[curElem],MaxPowerIters, SlackFactor,&PowerMethod_nIters,1.0f);
                            NewAlpha=1.0f/MaxEig;
                            EstChange=Xrms2*NewAlpha;
                            ItersToNextLineSearch=(MaxEstChangeToNextLineSearch/EstChange) / creal(ninneriter[curElem]);
                            // debug_printf(DP_DEBUG1,"ItersToNextLineSearch %d\n",ItersToNextLineSearch); 
                            CurTryRMS=-1.0f;
                            CurBeta=PowerMethod_nIters;
                        }

                        if(UseLineSearch) {
                            // f(x) = g(x) + h(x); g(x) is the data-consistency part; h(x) is the regularization
                            // tmpElems is Grad_g(x) (after real)
                            LastTryRMS=1e20;
                            for(i=0;i<MaxLineSearchIters;i++) {
                                md_zsmul(N, mapsDims[curElem], pElems[curElem], tmpElems[curElem], CurBeta);
                                md_zadd(N, mapsDims[curElem], pElems[curElem], pElems[curElem], Elems[curElem]);

                                // may apply proximal here to pElems
                                // Applies proximal on tmpElem into Elem, using pElem as temporary
                                // ApplyProximal(N mapDims,mapDimsR, Trg, Temp, Src, ...)
                                // md_copy(DIMS, mapDims, tmpElems2[curElem], Elems[curElem], CFL_SIZE);
                                // ApplyProximal(N, mapsDims[curElem], mapsDimsR[curElem],  tmpElems2[curElem], duElems[curElem], pElems[curElem],
                                //     CurElementType, creal(CurBeta),creal(ElemsLambda[curElem]), it_prox_ops[curElem], WrappingTrick[curElem], ElemsMin[curElem], ElemsMax[curElem],RandMaps[curElem],
                                //     &RelMapChangeWithAlphaAfterProx);
                                // md_copy(DIMS, mapDims, pElems[curElem], tmpElems2[curElem], CFL_SIZE);

                                operator_apply(sLuOps[curElem],N, XDims, ThisOne, N, mapsDims[curElem], pElems[curElem]);
                                md_zmul(DIMS, XDims, ThisOne, ThisOne, Others);
                                operator_apply(Aop->normal,N, XDims, ThisOne, N, XDims, ThisOne);
                                md_zsub(N, XDims, ThisOne, sig_adj, ThisOne); // r=AHsig-r
                                CurTryRMS=md_zrms(N, XDims, ThisOne);
                                EstChange=Xrms2*CurBeta;
                                debug_printf(DP_DEBUG2, "Line %d : iter %d Beta %6.2e CurTryRMS %7.4f%% EstChange %5.2e\n",curElem,i,CurBeta,100*CurTryRMS/rms_sig_adj,EstChange);
                                if(EstChange*CurBeta<MinEstChange) { break; }
                                if(CurTryRMS>LastTryRMS) { CurBeta/=BetaDecay; CurTryRMS=LastTryRMS; EstChange=Xrms2*CurBeta; break; }
                                LastTryRMS=CurTryRMS;
                                CurBeta*=BetaDecay;
                            }
                            ItersToNextLineSearch=(MaxEstChangeToNextLineSearch/(EstChange*FinalBetaFactor)) / creal(ninneriter[curElem]);
                            NewAlpha=CurBeta*FinalBetaFactor;
                        }
                        AlphaRatio=ElemsAlpha[curElem]/NewAlpha;
                        ElemsAlpha[curElem]=NewAlpha;

                        ItersToNextLineSearch=MAX(1,MIN(ItersToNextLineSearch,iter*ItersWithoutLineSearchAsRatioOfIter));
                        NextLineSearch[curElem]=iter+ MIN(MaxItersWithoutLineSearch,ItersToNextLineSearch);
                        debug_printf(DP_DEBUG1, "Line %2d : iter %5d:%2d Beta %6.2e CurTryRMS %7.4f%% EstChange %5.2e Alpha set to %5.2e r %5.2f\n",
                                            curElem,iter,i,CurBeta,100*CurTryRMS/rms_sig_adj,EstChange,NewAlpha,AlphaRatio);
                    }
                }
                // End Line search

                debug_printf(DP_DEBUG3, "Update curAlpha\n");
                CurAlpha=h*ElemsAlpha[curElem];

                
                debug_printf(DP_DEBUG3, "Time\n");
                time_t rawtime;
                struct tm * timeinfo;
                time ( &rawtime );
                timeinfo = localtime ( &rawtime );
                debug_printf(DP_DEBUG3, "Got time\n");

                long Level=DP_DEBUG2;
                if(inneriter==0) {
                    Level=DP_DEBUG1;
                    if( (iter%100) <=1) {
                        Level=DP_INFO;
                    }
                }

                debug_printf(DP_DEBUG3, "Apply alpha\n");
                // printf("CurAlpha %g + i%g\n",creal(CurAlpha),cimag(CurAlpha));
                md_zsmul(N, mapsDims[curElem], tmpElems[curElem], tmpElems[curElem], CurAlpha);
                md_zadd(N, mapsDims[curElem], tmpElems[curElem], tmpElems[curElem], Elems[curElem]);


                setCurRegulaizerCost(-1.0f);
                
                ApplyProximal(N, mapsDims[curElem], mapsDimsR[curElem],  Elems[curElem], pElems[curElem], tmpElems[curElem],
                    CurElementType, creal(CurAlpha),creal(ElemsLambda[curElem]), it_prox_ops[curElem], WrappingTrick[curElem], ElemsMin[curElem], ElemsMax[curElem],RandMaps[curElem],
                    &RelMapChangeWithAlphaAfterProx);

                CurRegCosts[curElem]=getCurRegulaizerCost();
                debug_printf(DP_DEBUG2, "Reg cost [%d] %f\n",curElem,CurRegCosts[curElem]);


    //             debug_printf(DP_DEBUG3, "Applying proximal %ld\n",curElem);
          //       // Apply proximal: Elems[curElem] -> Elems[curElem]
                // if(WrappingTrick[curElem] && (creal(ElemsLambda[curElem])>0) ) { // wrapping stuff for P
                //  debug_printf(DP_DEBUG2, "WrappingTrick\n");
                //  long rndI=floor(frand()*N_RAND_MAPS_FOR_P);

                //  md_clear(N, mapsDims[curElem], pElems[curElem], CFL_SIZE);
                //  md_zadd(N, mapsDims[curElem], pElems[curElem], pElems[curElem], RandMaps[curElem][rndI]);
                    
                    
                //  md_zexpj(N, mapsDims[curElem],pElems[curElem],pElems[curElem]);

    //                 md_zexpj(N, mapsDims[curElem],Elems[curElem],tmpElems[curElem]);
    //                 md_zmul(N, mapsDims[curElem],Elems[curElem],Elems[curElem],pElems[curElem]);
    //                 md_zarg(N, mapsDims[curElem],pElems[curElem],Elems[curElem]);                    

                //  // Here calling prox
                //  if(creal(ElemsLambda[curElem])>0) {
    //                     iter_op_p_call(it_prox_ops[curElem], creal(CurAlpha), (float*)Elems[curElem], (float*)pElems[curElem]);
                //  } else {
    //                     md_copy(DIMS, mapsDims[curElem], Elems[curElem], pElems[curElem], CFL_SIZE);
                //  }

                //  md_clear(N, mapsDims[curElem], pElems[curElem], CFL_SIZE);
                //  md_zsub(N, mapsDims[curElem], pElems[curElem], pElems[curElem], RandMaps[curElem][rndI]);

                //  md_zexpj(N, mapsDims[curElem],pElems[curElem],pElems[curElem]);
                //  md_zexpj(N, mapsDims[curElem],Elems[curElem],Elems[curElem]);
                //  md_zmul(N, mapsDims[curElem],Elems[curElem],Elems[curElem],pElems[curElem]);
                //  md_zarg(N, mapsDims[curElem],Elems[curElem],Elems[curElem]);
                //  debug_printf(DP_DEBUG2, "ok WrappingTrick\n");
                // } else { // No wrapping trick
                //  if(creal(ElemsLambda[curElem])>0) {
                //      iter_op_p_call(it_prox_ops[curElem], creal(CurAlpha), (float*)Elems[curElem], (float*)tmpElems[curElem]);
          //        } else { // No proximal
          //            md_copy(DIMS, mapsDims[curElem], Elems[curElem], tmpElems[curElem], CFL_SIZE); // ok
          //        }
          //       }
          //       if(CurElementType==1) { // abs for M
          //        md_zabs(N, mapsDims[curElem], Elems[curElem], Elems[curElem]);
          //       }
          //       if(CurElementType==4) { // max with 5 for T2*
          //        debug_printf(DP_DEBUG3, "Max op\n");
          //        md_zabs(N, mapsDims[curElem], Elems[curElem], Elems[curElem]);
          //       }

    //             md_min(N, mapsDimsR[curElem], Elems[curElem], ElemsMax[curElem], Elems[curElem]);
    //             md_zmax(N, mapsDimsR[curElem], Elems[curElem], ElemsMin[curElem], Elems[curElem]);

                // md_zsub(N, mapsDims[curElem], tmpElems[curElem], tmpElems[curElem], Elems[curElem]);
                // RelMapChangeWithAlphaAfterProx=md_zrms(N, mapsDims[curElem], tmpElems[curElem])/md_zrms(N, mapsDims[curElem], Elems[curElem]);
                



                TotalTime_d += (timestamp()-Before_d);
                
                // if(inneriter==0) 
                {
                    float TimePerIter=(timestamp()-IterStartTime)*1000.0/(iter+1);
                    float TimePerIter_Normal=TotalTime_Normal*1000.0/(iter+1);
                    float TimePerIter_a=TotalTime_a*1000.0/(iter+1);
                    float TimePerIter_b=TotalTime_b*1000.0/(iter+1);
                    float TimePerIter_c=TotalTime_c*1000.0/(iter+1);
                    float TimePerIter_d=TotalTime_d*1000.0/(iter+1);
                    if(hasGT) {
                        debug_printf(Level, "%s[%02ld:%02ld:%02ld,%4.0f,%4.0fms] ",Msg,timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec,TimePerIter,TimePerIter_Normal);
                        debug_printf(Level, "Iter %03ld:%ld:%ld Cur rrms %5.2f%% GT-rms %5.2f%% Rel map ch w/alpha %8.2e, p %8.2e\n",iter,curElem,inneriter,100.0f*curerr/rms_sig_adj,100.0f*Xrms,Xrms2*creal(CurAlpha),RelMapChangeWithAlphaAfterProx);
                    } else {
                        debug_printf(Level, "%s[%02ld:%02ld:%02ld,%4.0f,N%4.0f, %4.0f, %4.0f, %4.0f, %4.0fms] ",Msg,timeinfo->tm_hour, timeinfo->tm_min, timeinfo->tm_sec,TimePerIter,TimePerIter_Normal,TimePerIter_a,TimePerIter_b,TimePerIter_c,TimePerIter_d);
                        debug_printf(Level, "Iter % 3ld:%ld:%ld Cur rrms %5.2f%% Rel map ch w/alpha %8.2e, p %8.2e\n",iter,curElem,inneriter,100.0f*curerr/rms_sig_adj,Xrms2*creal(CurAlpha),RelMapChangeWithAlphaAfterProx);
                    }
                }

                // toimg_stack(argv[2], false, true, false, 1, 0, 750, dims, data);

                debug_printf(DP_DEBUG3, "Finished this inner iteration\n");
                TotalTime_b += (timestamp()-Before_b);
            } // for innerIter

            if(StopIterations) {
                break;
            }
            // end_time = timestamp();
            // printf("b %f\n",end_time-cur_time1);
        } // for curElem

        TotalTime_a += (timestamp()-Before_a);

        if(StopIterations) {
            break;
        }

        if(dohogwild) {
            k = k + 1;
            if (k == K) {
                k = 0;
                K = K * 2;
                h = h / 2;
            }
        }

        double cur_time = timestamp();
        double TimeSinceLastSave=cur_time-last_save_time;
        if(TimeSinceLastSave>TimeBetweenSaves) {
            last_save_time = timestamp();
            for(curElem=0;curElem<nElements;curElem++) {
                if(gpu) {
                    md_copy(DIMS, mapsDims[curElem], Elemsa[curElem], Elems[curElem], CFL_SIZE);
                } else {
                    md_copy(DIMS, mapsDims[curElem], pElems[curElem], Elems[curElem], CFL_SIZE);
                }
                // No need to unmap and remap?
                // unmap_cfl(N, mapsDims[curElem], Elemsa[curElem]); 
            }
            debug_printf(DP_INFO,"Saved maps\n");
            for(curElem=0;curElem<nElements;curElem++) {
                // No need to unmap and remap?
                // snprintf(FNbufx, MAX_FN_LEN, "Elem%ld", curElem);
                // PRINT_TO_FNBUF(FNbufx);
                // debug_printf(DP_INFO,"Creating maps %ld\n",curElem);
                // Elemsa[curElem]=create_cfl(FNbuf, N, mapsDims[curElem]);

                // if(gpu) {
                //  Elems[curElem]=md_gpu_move(DIMS, mapsDims[curElem], Elemsa[curElem], CFL_SIZE); 
                // } else {
                //  Elems[curElem]=Elemsa[curElem];
                // }
                if(!gpu) {
                    Elems[curElem]=Elemsa[curElem];
                    md_copy(DIMS, mapsDims[curElem], Elems[curElem], pElems[curElem], CFL_SIZE);
                }
            }
        }

        if(IntermediateSaveIter>0) {
            if( (iter>1) && (iter%IntermediateSaveIter)==0 ) {
                complex float* Elemsb[MAX_MAPS];
                for(i=0;i<nElements;i++) {
                    snprintf(FNbufx, MAX_FN_LEN, "Elem%ld_iter%ld", i,iter);
                    PRINT_TO_FNBUFOUT(FNbufx);
                    Elemsb[i]=create_cfl(FNbuf, N, mapsDims[i]);
                    md_copy(DIMS, mapsDims[i], Elemsb[i], Elems[i], CFL_SIZE);

                    unmap_cfl(N, mapsDims[i], Elemsb[i]);
                }
                debug_printf(DP_INFO,"Saved intermediate maps for iter %ld\n",iter);
            }
        }
    } // for Iter

    ErrOut[ContIter++]=-1;

    // debug_printf(DP_INFO,"Finished iters\n");
    float TotalTime=timestamp()-Start_time;
    debug_printf(DP_INFO,"Finished iters, it took %f seconds for %d iters, avg %.2fms\n",TotalTime,ContIter,1000*TotalTime/ContIter);
    
    unmap_cfl(N, ErrOutDims, ErrOut);

    if(!DoNotSave) {
        for(curElem=0;curElem<nElements;curElem++) {
            if(gpu) {
                md_copy(DIMS, mapsDims[curElem], Elemsa[curElem], Elems[curElem], CFL_SIZE);
            }
            unmap_cfl(N, mapsDims[curElem], Elemsa[curElem]); }
        debug_printf(DP_INFO,"Saved maps\n");
    }

    if(gpu) {
        md_copy(DIMS, XDims, XDimsOuta, XDimsOut, CFL_SIZE);
        md_copy(DIMS, mapsDims[0], mDimsOuta, mDimsOut, CFL_SIZE);
    }
    unmap_cfl(N, XDims, XDimsOuta);
    unmap_cfl(N, mapsDims[0], mDimsOuta);
    // debug_printf(DP_INFO,"aaa\n");

    for(curElem=0;curElem<nElements;curElem++) {
        md_free(duElems[curElem]);
        md_free(tmpElems[curElem]);
        md_free(pElems[curElem]);
    }

    for(i=0;i<nElements;i++) {
        for(j=0;j<N_AUX_MAPS;j++) {
            if(AuxMaps[i][j]!=NULL) {
                md_free(AuxMaps[i][j]);
            }
        }
    }

    for(i=0;i<nElements;i++) {
        for(j=0;j<N_RAND_MAPS_FOR_P;j++) {
            if(RandMaps[i][j]!=NULL) {
                md_free(RandMaps[i][j]);
            }
        }
    }
    // debug_printf(DP_INFO,"bbb\n");
    md_free(Others);
    md_free(ThisOne);
    md_free(r);

    if(hasGT) {
        unmap_cfl(N, XDims, GTa);
    }
    unmap_cfl(N, XDims, sig_adja);
    
    if(gpu) {
        if(hasGT) {
            md_free(GT); }
        md_free(sig_adj);
    }

    debug_printf(DP_INFO,"Preparing out\n");
    complex float* out = create_cfl(argv[num_args], DIMS, mapsDims[0]);
    md_clear(DIMS, mapsDims[0], out, CFL_SIZE);
    unmap_cfl(DIMS, mapsDims[0], out);

    ClearReadScriptFiles(&argv[4],num_args-4);
    FreeLinops();
    FreeOutLinops();
    
    debug_printf(DP_INFO,"splitProx done\n");

    return 0;
}
