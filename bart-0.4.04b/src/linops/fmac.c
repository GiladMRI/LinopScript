/* Copyright 2016-2018. Martin Uecker.
 * All rights reserved. Use of this source code is governed by
 * a BSD-style license which can be found in the LICENSE file.
 *
 * Authors:
 * 2016-2018 Martin Uecker <martin.uecker@med.uni-goettingen.de>
 */

#include <string.h>
#include <complex.h>
#include <assert.h>
#include <math.h>
#include <stdbool.h>

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/ops.h"
#ifdef USE_CUDA
#include "num/gpuops.h"
#endif
//#include "num/iovec.h"

#include "linops/linop.h"

#include "misc/misc.h"

#include "fmac.h"

#include "misc/debug.h"






struct fmac_data {

	INTERFACE(linop_data_t);

	unsigned int N;
	long *dims;

	long *idims;
	long *istrs;

	long *odims;
	long *ostrs;

	long *tdims;
	long *tstrs;

	const complex float* tensor;
#ifdef USE_CUDA
	const complex float* gpu_tensor;
#endif
};

static DEF_TYPEID(fmac_data);

#ifdef USE_CUDA
static const complex float* get_tensor(const struct fmac_data* data, bool gpu)
{
	const complex float* tensor = data->tensor;

	if (gpu) {

		if (NULL == data->gpu_tensor)
			((struct fmac_data*)data)->gpu_tensor = md_gpu_move(data->N, data->tdims, data->tensor, CFL_SIZE);

		tensor = data->gpu_tensor;
	}

	return tensor;
}
#endif



static void fmac_free_data(const linop_data_t* _data)
{
        auto data = CAST_DOWN(fmac_data, _data);

#ifdef USE_CUDA
	if (NULL != data->gpu_tensor)
		md_free((void*)data->gpu_tensor);
#endif

	xfree(data->dims);
	xfree(data->idims);
	xfree(data->istrs);
	xfree(data->odims);
	xfree(data->ostrs);
	xfree(data->tdims);
	xfree(data->tstrs);

	xfree(data);
}


static void fmac_apply(const linop_data_t* _data, complex float* dst, const complex float* src)
{
        auto data = CAST_DOWN(fmac_data, _data);

#ifdef USE_CUDA
	const complex float* tensor = get_tensor(data, cuda_ondevice(src));
#else
	const complex float* tensor = data->tensor;
#endif

	md_clear2(data->N, data->odims, data->ostrs, dst, CFL_SIZE);
	md_zfmac2(data->N, data->dims, data->ostrs, dst, data->istrs, src, data->tstrs, tensor);
}

static void fmac_adjoint(const linop_data_t* _data, complex float* dst, const complex float* src)
{
        auto data = CAST_DOWN(fmac_data, _data);

#ifdef USE_CUDA
	const complex float* tensor = get_tensor(data, cuda_ondevice(src));
#else
	const complex float* tensor = data->tensor;
#endif

	md_clear2(data->N, data->idims, data->istrs, dst, CFL_SIZE);
	md_zfmacc2(data->N, data->dims, data->istrs, dst, data->ostrs, src, data->tstrs, tensor);
}

const struct linop_s* linop_fmac_create(unsigned int N, const long dims[N], 
		unsigned int oflags, unsigned int iflags, unsigned int tflags, const complex float* tensor)
{
	PTR_ALLOC(struct fmac_data, data);
	SET_TYPEID(fmac_data, data);

	data->N = N;

	data->dims = *TYPE_ALLOC(long[N]);
	md_copy_dims(N, data->dims, dims);

	data->idims = *TYPE_ALLOC(long[N]);
	data->istrs = *TYPE_ALLOC(long[N]);

	md_select_dims(N, ~iflags, data->idims, dims);
	md_calc_strides(N, data->istrs, data->idims, CFL_SIZE);

	data->odims = *TYPE_ALLOC(long[N]);
	data->ostrs = *TYPE_ALLOC(long[N]);

	md_select_dims(N, ~oflags, data->odims, dims);
	md_calc_strides(N, data->ostrs, data->odims, CFL_SIZE);

	data->tstrs = *TYPE_ALLOC(long[N]);
	data->tdims = *TYPE_ALLOC(long[N]);

	md_select_dims(N, ~tflags, data->tdims, dims);
	md_calc_strides(N, data->tstrs, data->tdims, CFL_SIZE);

	data->tensor = tensor;
#ifdef USE_CUDA
	data->gpu_tensor = NULL;
#endif

	long odims[N];
	md_copy_dims(N, odims, data->odims);

	long idims[N];
	md_copy_dims(N, idims, data->idims);

	return linop_create(N, odims, N, idims,
			CAST_UP(PTR_PASS(data)), fmac_apply, fmac_adjoint, NULL,
			NULL, fmac_free_data);
}


































// Now PartitionDim
struct PartitionDim_data {

	INTERFACE(linop_data_t);

	unsigned int N;
	long *idims;
	long *istrs;

	long *odims;
	long *ostrs;
};

static DEF_TYPEID(PartitionDim_data);



static void PartitionDim_free_data(const linop_data_t* _data)
{
        auto data = CAST_DOWN(PartitionDim_data, _data);

	xfree(data->idims);
	xfree(data->istrs);
	xfree(data->odims);
	xfree(data->ostrs);

	xfree(data);
}


static void PartitionDim_apply(const linop_data_t* _data, complex float* dst, const complex float* src)
{
        auto data = CAST_DOWN(PartitionDim_data, _data);

	// void md_copy2(unsigned int D, const long dim[D], const long ostr[D], void* optr, const long istr[D], const void* iptr, size_t size)

	md_copy(data->N, data->odims, dst, src, CFL_SIZE);
	// md_copy2(data->N, data->odims, data->ostrs, dst, data->istrs, src, CFL_SIZE);

	// md_clear2(data->N, data->odims, data->ostrs, dst, CFL_SIZE);
	// md_zfmac2(data->N, data->dims, data->ostrs, dst, data->istrs, src, data->tstrs, tensor);
}

static void PartitionDim_adjoint(const linop_data_t* _data, complex float* dst, const complex float* src)
{
        auto data = CAST_DOWN(PartitionDim_data, _data);

	md_copy(data->N, data->odims, dst, src, CFL_SIZE);
	// md_copy2(data->N, data->idims, data->istrs, dst, data->ostrs, src, CFL_SIZE);

	// md_clear2(data->N, data->idims, data->istrs, dst, CFL_SIZE);
	// md_zfmacc2(data->N, data->dims, data->istrs, dst, data->ostrs, src, data->tstrs, tensor);
}

// static void PartitionDim_normal(const linop_data_t* _data, complex float* dst, const complex float* src)
// {
//         auto data = CAST_DOWN(PartitionDim_data, _data);

// 	md_copy2(data->N, data->idims, data->istrs, dst, data->istrs, src, CFL_SIZE);

// 	// md_clear2(data->N, data->idims, data->istrs, dst, CFL_SIZE);
// 	// md_zfmacc2(data->N, data->dims, data->istrs, dst, data->ostrs, src, data->tstrs, tensor);
// }

const struct linop_s* linop_PartitionDim_create(unsigned int N, const long dims[N], 
		const long Dim1, const long Dim2, const long K)
{
	PTR_ALLOC(struct PartitionDim_data, data);
	SET_TYPEID(PartitionDim_data, data);

	debug_printf(DP_INFO,"linop_PartitionDim_create %ld %ld %ld\n",Dim1,Dim2,K);
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
	md_calc_strides(N, data->ostrs, data->odims, CFL_SIZE);

	data->odims[Dim1]=data->odims[Dim1]/K;
	data->odims[Dim2]=data->odims[Dim2]*K;

	data->ostrs[Dim2]=data->istrs[Dim1];
	data->ostrs[Dim2]=data->istrs[Dim1]*data->odims[Dim1];

	debug_printf(DP_INFO,"odims: ");
    debug_print_dims(DP_INFO,N,data->odims);

	debug_printf(DP_INFO,"istrs: ");
    debug_print_dims(DP_INFO,N,data->istrs);
	debug_printf(DP_INFO,"ostrs: ");
    debug_print_dims(DP_INFO,N,data->ostrs);

	long odims[N];
	md_copy_dims(N, odims, data->odims);

	long idims[N];
	md_copy_dims(N, idims, data->idims);

	return linop_create(N, odims, N, idims,
			CAST_UP(PTR_PASS(data)), PartitionDim_apply, PartitionDim_adjoint, NULL,
			NULL, PartitionDim_free_data);
}