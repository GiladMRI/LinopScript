/* Copyright 2013-2014. The Regents of the University of California.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2012 Martin Uecker <uecker@eecs.berkeley.edu>
 * 2014 Frank Ong <frankong@berkeley.edu>
 */

#include <complex.h>
#include <assert.h>

#include "num/multind.h"
#include "num/flpmath.h"

#include "misc/debug.h"

#include "linops/linop.h"

#include "casorati.h"

#include <math.h>
#include <stdbool.h>

#include "num/ops.h"
#ifdef USE_CUDA
#include "num/gpuops.h"
#endif

#include "misc/misc.h"



// ggg removed static
// static 
void calc_casorati_geom(unsigned int N, long dimc[2 * N], long str2[2 * N], const long dimk[N], const long dim[N], const long str[N])
{
	for (unsigned int i = 0; i < N; i++) {

		assert(dim[i] >= dimk[i]);

		dimc[i + 0] = dim[i] - dimk[i] + 1;	// number of shifted blocks
		dimc[i + N] = dimk[i];			// size of blocks

		str2[i + N] = str[i];			// by having the same strides
		str2[i + 0] = str[i];			// we can address overlapping blocks
	}
}


void casorati_dims(unsigned int N, long odim[2], const long dimk[N], const long dim[N])
{
	long dimc[2 * N];

	for (unsigned int i = 0; i < N; i++) {

		assert(dim[i] >= dimk[i]);

		dimc[i + 0] = dim[i] - dimk[i] + 1;	// number of shifted blocks
		dimc[i + N] = dimk[i];			// size of blocks
	}

	odim[0] = md_calc_size(N, dimc + 0);
	odim[1] = md_calc_size(N, dimc + N);
}


void casorati_matrix(unsigned int N, const long dimk[N], const long odim[2], complex float* optr, const long dim[N], const long str[N], const complex float* iptr)
{
	long str2[2 * N];
	long strc[2 * N];
	long dimc[2 * N];

	calc_casorati_geom(N, dimc, str2, dimk, dim, str);

	assert(odim[0] == md_calc_size(N, dimc + 0));	// all shifts are collapsed
	assert(odim[1] == md_calc_size(N, dimc + N));	// linearized size of a block

	debug_printf(DP_INFO,"dimc:");
	debug_print_dims(DP_INFO,2 * N,dimc);

	md_calc_strides(2 * N, strc, dimc, CFL_SIZE);
	md_copy2(2 * N, dimc, strc, optr, str2, iptr, CFL_SIZE);
}



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






static void calc_basorati_geom(unsigned int N, long dimc[2 * N], long str2[2 * N], const long dimk[N], const long dim[N], const long str[N])
{
	for (unsigned int i = 0; i < N; i++) {

		dimc[i + 0] = dimk[i];			// size of blocks
		dimc[i + N] = dim[i] / dimk[i];	// number of shifted blocks

		str2[i + 0] = str[i];
		str2[i + N] = str[i] * dimk[i];
	}
}


void basorati_dims(unsigned int N, long odim[2], const long dimk[N], const long dim[N])
{
	long dimc[2 * N];

	for (unsigned int i = 0; i < N; i++) {

		assert(0 == dim[i] % dimk[i]);

		dimc[i + 0] = dimk[i];		// size of blocks
		dimc[i + N] = dim[i] / dimk[i];	// number of shifted blocks
	}

	odim[0] = md_calc_size(N, dimc + 0);
	odim[1] = md_calc_size(N, dimc + N);
}


void basorati_matrix(unsigned int N, const long dimk[N], const long odim[2], complex float* optr, const long dim[N], const long str[N], const complex float* iptr)
{
	long str2[2 * N];
	long strc[2 * N];
	long dimc[2 * N];

	calc_basorati_geom(N, dimc, str2, dimk, dim, str);

	assert(odim[0] == md_calc_size(N, dimc + 0));	// all shifts are collapsed
	assert(odim[1] == md_calc_size(N, dimc + N));	// linearized size of a block

	md_calc_strides(2 * N, strc, dimc, CFL_SIZE);
	md_copy2(2 * N, dimc, strc, optr, str2, iptr, CFL_SIZE);
}



void basorati_matrixH(unsigned int N, const long dimk[N], const long dim[N], const long str[N], complex float* optr, const long odim[2], const complex float* iptr)
{
	long str2[2 * N];
	long strc[2 * N];
	long dimc[2 * N];

	calc_basorati_geom(N, dimc, str2, dimk, dim, str);

	assert(odim[0] == md_calc_size(N, dimc));
	assert(odim[1] == md_calc_size(N, dimc + N));

	md_calc_strides(2 * N, strc, dimc, CFL_SIZE);
	md_copy2(2 * N, dimc, str2, optr, strc, iptr, CFL_SIZE);
}



void gcasorati_dims(unsigned int N, long odim[2], const long dimk[N], const long dim[N])
{
	long dimc[2 * N];

	for (unsigned int i = 0; i < N; i++) {

		assert(dim[i] >= dimk[i]);

		dimc[i + 0] = dim[i] - dimk[i] + 1;	// number of shifted blocks
		dimc[i + N] = dimk[i];			// size of blocks
	}

	odim[0] = md_calc_size(N, dimc + 0);
	odim[1] = md_calc_size(N, dimc + N);
}


void gcasorati_matrix(unsigned int N, const long dimk[N], const long odim[2], complex float* optr, const long dim[N], const long str[N], const complex float* iptr)
{
	long str2[2 * N];
	long strc[2 * N];
	long dimc[2 * N];

	calc_casorati_geom(N, dimc, str2, dimk, dim, str);

	assert(odim[0] == md_calc_size(N, dimc + 0));	// all shifts are collapsed
	assert(odim[1] == md_calc_size(N, dimc + N));	// linearized size of a block

	debug_printf(DP_INFO,"dimc:");
	debug_print_dims(DP_INFO,2 * N,dimc);

	md_calc_strides(2 * N, strc, dimc, CFL_SIZE);
	md_copy2(2 * N, dimc, strc, optr, str2, iptr, CFL_SIZE);
}
























// Now Hankel
struct Hankel_data {

	INTERFACE(linop_data_t);

	unsigned int N;
	long *idims;
	long *istrs;

	long *odims;
	long *ostrs;

	long *strc;
};

static DEF_TYPEID(Hankel_data);



static void Hankel_free_data(const linop_data_t* _data)
{
        auto data = CAST_DOWN(Hankel_data, _data);

	xfree(data->idims);
	xfree(data->istrs);
	xfree(data->odims);
	xfree(data->ostrs);

	xfree(data->strc);

	xfree(data);
}


static void Hankel_apply(const linop_data_t* _data, complex float* dst, const complex float* src)
{
        auto data = CAST_DOWN(Hankel_data, _data);

	// debug_printf(DP_INFO, "Hankel_apply start\n");

	md_copy2(data->N, data->odims, data->strc, dst, data->ostrs, src, CFL_SIZE);

	// debug_printf(DP_INFO, "Hankel_apply end\n");

	// md_copy(data->N, data->odims, dst, src, CFL_SIZE);
}

static void Hankel_adjoint(const linop_data_t* _data, complex float* dst, const complex float* src)
{
        auto data = CAST_DOWN(Hankel_data, _data);

	// debug_printf(DP_INFO, "Hankel_adj start\n");
	// md_copy(data->N, data->odims, dst, src, CFL_SIZE);
	md_clear(data->N, data->idims, dst, CFL_SIZE);
	md_zadd2(data->N, data->odims, data->ostrs, dst, data->ostrs, dst, data->strc, src);

	// debug_printf(DP_INFO, "Hankel_adj end\n");
}

const struct linop_s* linop_Hankel_create(unsigned int N, const long dims[N], 
		const long Dim1, const long Dim2, const long K)
{
	PTR_ALLOC(struct Hankel_data, data);
	SET_TYPEID(Hankel_data, data);

	debug_printf(DP_INFO,"linop_Hankel_create %ld %ld %ld\n",Dim1,Dim2,K);
	debug_printf(DP_INFO,"dims: ");
    debug_print_dims(DP_INFO,N,dims);


	data->N = N;

	data->idims = *TYPE_ALLOC(long[N]);
	data->istrs = *TYPE_ALLOC(long[N]);

	md_copy_dims(N, (long*)data->idims, dims);
	md_calc_strides(N, data->istrs, data->idims, CFL_SIZE);

	data->odims = *TYPE_ALLOC(long[N]);
	data->ostrs = *TYPE_ALLOC(long[N]);

	md_copy_dims(N, (long*)data->odims, dims);
	data->odims[Dim1]=data->idims[Dim1]-K+1;
    data->odims[Dim2]=K;

	md_copy_strides(N, data->ostrs, data->istrs);
    data->ostrs[Dim1]=data->istrs[Dim1];
    data->ostrs[Dim2]=data->istrs[Dim1];

	data->strc = *TYPE_ALLOC(long[N]);
	md_calc_strides(N, data->strc, data->odims, CFL_SIZE);

	long odims[N];
	md_copy_dims(N, odims, data->odims);

	long idims[N];
	md_copy_dims(N, idims, data->idims);

	return linop_create(N, odims, N, idims,
			CAST_UP(PTR_PASS(data)), Hankel_apply, Hankel_adjoint, NULL,
			NULL, Hankel_free_data);
}
