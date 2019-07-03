/* Copyright 2017. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by 
 * a BSD-style license which can be found in the LICENSE file.
 * 
 * Authors:
 * 2017 Jon Tamir <jtamir@eecs.berkeley.edu>
 */

#include <stdlib.h>
#include <assert.h>
#include <complex.h>
#include <stdio.h>

#include "num/multind.h"
#include "num/casorati.h"
#include "num/init.h"

#include "misc/mmio.h"
#include "misc/misc.h"
#include "misc/opts.h"

#include "misc/debug.h"

#ifndef DIMS
#define DIMS 16
#endif

#ifndef CFL_SIZE
#define CFL_SIZE	sizeof(complex float)
#endif

static const char usage_str[] = "dim1 kern1 dim2 <input for adjoint> <input> <output>";
static const char help_str[] = "Hankel on dim1 with Hankelization kern1 into dim2.\n";


int main_hankel(int argc, char* argv[])
{
    bool Normal = false;
    bool Adj = false;
    const char* adj_file = NULL;
    
	const struct opt_s opts[] = {
		OPT_SET('N', &Normal, "Apply Forward+adjoint"),
        OPT_SET('a', &Adj, "Apply adjoint"),
    };	

    cmdline(&argc, argv, 2, 100, usage_str, help_str, ARRAY_SIZE(opts), opts);
	int num_args = argc - 1;
	
    // printf("main_hankel\n");
    // printf("%s\n",argv[0]); // "hankel"
    // printf("%s\n",argv[1]); // script file
    // printf("%s\n",argv[2]); // output
    // printf("%s\n",argv[3]); // input
    // printf("BBB\n");

	num_init();

	int count = argc - 3;
	
    if(Adj) {
        assert(count == 4);
    } else {
	    assert(count == 3);
    }

	long idims[DIMS];
	long kdims[DIMS];
	long odims[DIMS];

	complex float* idata = load_cfl(argv[argc - 2], DIMS, idims);

	md_copy_dims(DIMS, kdims, idims);

    unsigned int dim1=atoi(argv[1]);
    unsigned int kern1=atoi(argv[2]);
    unsigned int dim2=atoi(argv[3]);

    debug_printf(DP_INFO,"Hankel with %d %d %d\n",dim1,kern1,dim2);

    debug_printf(DP_INFO,"idims:");
	debug_print_dims(DP_INFO,DIMS,idims);

    assert(idims[dim2] == 1);

    long istrs[DIMS];
	md_calc_strides(DIMS, istrs, idims, CFL_SIZE);

    md_copy_dims(DIMS, odims, idims);
    odims[dim1]=idims[dim1]-kern1+1;
    odims[dim2]=kern1;

    long ostrs[DIMS];

    md_copy_strides(DIMS, ostrs, istrs);
    ostrs[dim1]=istrs[dim1];
    ostrs[dim2]=istrs[dim1];

    complex float* odata;
    if(Normal) {
        debug_printf(DP_INFO,"Normal!\n");
        odata = md_alloc(DIMS, odims, CFL_SIZE);
    } else {
        if(!Adj) {
            odata = create_cfl(argv[argc - 1], DIMS, odims);
        }
    }

    
    long strc[DIMS];
    md_calc_strides(DIMS, strc, odims, CFL_SIZE);

    // debug_printf(DP_INFO,"bb!\n");
    // debug_printf(DP_INFO,"strc:");
	// debug_print_dims(DP_INFO,DIMS,strc);
    // debug_printf(DP_INFO,"ostrs:");
	// debug_print_dims(DP_INFO,DIMS,ostrs);

    if(Normal) {
        debug_printf(DP_INFO,"Normal a!\n");
        md_copy2(DIMS, odims, strc, odata, ostrs, idata, CFL_SIZE);

        unmap_cfl(DIMS, idims, idata);

        complex float* Nodata;
        Nodata = create_cfl(argv[argc - 1], DIMS, idims);
        md_clear(DIMS, idims, Nodata, CFL_SIZE);
        // void md_zadd2(unsigned int D, const long dims[D], const long ostr[D], complex float* optr, const long istr1[D], const complex float* iptr1, const long istr2[D], const complex float* iptr2)
        md_zadd2(DIMS, odims, ostrs, Nodata, ostrs, Nodata, strc, odata);
        unmap_cfl(DIMS, idims, Nodata);

        md_free(odata);
    } else {
        if(Adj) {
            long adims[DIMS];
            complex float* adata = load_cfl(argv[argc - 3], DIMS, adims); // adims should be equal to odims

            // debug_printf(DP_INFO,"adims:");
	        // debug_print_dims(DP_INFO,DIMS,adims);
            
            complex float* Nodata;
            Nodata = create_cfl(argv[argc - 1], DIMS, idims);
            md_clear(DIMS, idims, Nodata, CFL_SIZE);
            // void md_zadd2(unsigned int D, const long dims[D], const long ostr[D], complex float* optr, const long istr1[D], const complex float* iptr1, const long istr2[D], const complex float* iptr2)
            md_zadd2(DIMS, odims, ostrs, Nodata, ostrs, Nodata, strc, adata);
            unmap_cfl(DIMS, idims, Nodata);

            // md_free(odata);
            unmap_cfl(DIMS, adims, adata);
        } else {
            md_copy2(DIMS, odims, strc, odata, ostrs, idata, CFL_SIZE);

            unmap_cfl(DIMS, idims, idata);

            unmap_cfl(DIMS, odims, odata);
        }
    }

    
/*
void casorati_matrixH(unsigned int N, const long dimk[N], const long dim[N], const long str[N], complex float* optr, const long odim[2], const complex float* iptr)
{
	long str2[2 * N];
	long strc[2 * N];
	long dimc[2 * N];

	calc_casorati_geom(N, dimc, str2, dimk, dim, str);

	assert(odim[0] == md_calc_size(N, dimc));
	assert(odim[1] == md_calc_size(N, dimc + N));

	md_clear(N, dim, optr, CFL_SIZE);

	md_calc_strides(2 * N, strc, dimc, CFL_SIZE);
	md_zadd2(2 * N, dimc, str2, optr, str2, optr, strc, iptr);
}
*/

/*
	for (int i = 0; i < count; i += 2) {

		unsigned int kdim = atoi(argv[i + 1]);
		unsigned int ksize = atoi(argv[i + 2]);

		assert(kdim < DIMS);
		assert(ksize >= 1);

		kdims[kdim] = ksize;
	}


    // ggg removed
    // casorati_dims(DIMS, odims, kdims, idims);
	// complex float* odata = create_cfl(argv[argc - 1], 2, odims);

	long istrs[DIMS];
	md_calc_strides(DIMS, istrs, idims, CFL_SIZE);

    // added
    long str2[2 * DIMS];
	long strc[2 * DIMS];
	long dimc[2 * DIMS];

    debug_printf(DP_INFO,"idims:");
	debug_print_dims(DP_INFO,2 * DIMS,idims);
    debug_printf(DP_INFO,"istrs:");
	debug_print_dims(DP_INFO,2 * DIMS,istrs);
    debug_printf(DP_INFO,"kdims:");
	debug_print_dims(DP_INFO,DIMS,kdims);

	calc_casorati_geom(DIMS, dimc, str2, kdims, idims, istrs);

    debug_printf(DP_INFO,"dimc:");
	debug_print_dims(DP_INFO,DIMS*2,dimc);
    debug_printf(DP_INFO,"str2:");
	debug_print_dims(DP_INFO,DIMS*2,str2);

    complex float* odata = create_cfl(argv[argc - 1], 2*DIMS, dimc);

    md_calc_strides(2 * DIMS, strc, dimc, CFL_SIZE);

    debug_printf(DP_INFO,"strc:");
	debug_print_dims(DP_INFO,DIMS*2,strc);

	md_copy2(2 * DIMS, dimc, strc, odata, str2, idata, CFL_SIZE);
    // casorati_matrix(DIMS, kdims, dimc, odata, idims, istrs, idata);

    // ggg removed
	// casorati_matrix(DIMS, kdims, odims, odata, idims, istrs, idata);

	unmap_cfl(DIMS, idims, idata);
    // ggg changed
	// unmap_cfl(2, odims, odata);
    unmap_cfl(2*DIMS, dimc, odata);*/

	return 0;
}
