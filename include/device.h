/*****************************************************************************
 AUTHOR:
 Jedda Boyle

 CONTAINS:
 GPUEnviroment class that provides a wrapper for OpenCL components, and 
 provides an iterface to the GPU kernels.

 NOTES:

  *****************************************************************************/

#ifndef GPU_ENVIROMENT_H
#define GPU_ENVIROMENT_H

#include <vector>
#include <CL/cl.hpp>


#include "timing.h"

class GPUEnviroment {
public:
	
	// GPU kernels which this class provides a wrapper for.
	cl::Kernel mask;
	cl::Kernel grubb;
	cl::Kernel row_medians;
	cl::Kernel mask_rows;
	cl::Kernel constant_row_mask;
	cl::Kernel transpose;
	cl::Kernel mad_rows;
	cl::Kernel edge_threshold;

	// OpenCL enviroemtn variables.
	cl::Program program;
	cl::Context context;
	cl::CommandQueue queue;
	std::vector<cl::Device> devices;	

	// cl_int used to get error reporting from OpenCL.
	cl_int error_code;

	// Setup context, queue, devices and kernels.
	GPUEnviroment (void);

	// Util functions for handling GPU memory.
	cl::Buffer InitBuffer(const cl_mem_flags mem_flag, const size_t size);

    INIT_MARK(mark);
	INIT_TIMER(transpose_timer);
	INIT_TIMER(medians_timer);
	INIT_TIMER(grubb_timer);
	INIT_TIMER(mask_rows_timer);
	INIT_TIMER(mask_timer);
	INIT_TIMER(mad_timer);
	INIT_TIMER(edge_timer);
	INIT_TIMER(const_mask_rows_timer);


	void PrintKernelTimers() {
		PRINT_TIMER(transpose_timer);
		PRINT_TIMER(medians_timer);
		PRINT_TIMER(grubb_timer);
		PRINT_TIMER(mask_rows_timer);
		PRINT_TIMER(mask_timer);
		PRINT_TIMER(mad_timer);
		PRINT_TIMER(edge_timer);
		PRINT_TIMER(const_mask_rows_timer);


	}

	void WriteToBuffer(void* host_mem, cl::Buffer& buffer, const size_t size);

	void ReadFromBuffer(void* host_mem, cl::Buffer& buffer, const size_t size);

	void CopyBuffer (const cl::Buffer& src, cl::Buffer& dest, size_t size);

	//void Mask(const cl::Buffer& d_out, cl::Buffer& d_in, cl::Buffer& d_mask, uint8_t mask_value, size_t m, size_t n, size_t local_size_m, size_t local_size_n);
	void Mask(const cl::Buffer& d_out, cl::Buffer& d_in, cl::Buffer& d_mask, cl::Buffer& freq_medians, size_t m, size_t n, size_t local_size_m, size_t local_size_n);

	void MaskRows(const cl::Buffer& data, cl::Buffer& mask, cl::Buffer& medians, size_t m, size_t n, size_t local_size);
	void ConstantRowMask(const cl::Buffer& data, cl::Buffer& mask, size_t m, size_t n, size_t local_size);

	void Transpose( cl::Buffer& d_out, cl::Buffer& d_in, size_t m, size_t n, size_t local_size_m, size_t local_size_n);
	//void Transpose(cl::Buffer& d_out, cl::Buffer& d_in, size_t m, size_t n, size_t tile_dim, size_t local_size_m);

	//void Transpose2( cl::Buffer& d_out, cl::Buffer& d_in, size_t m, size_t n, size_t tile_dim);

	void EdgeThreshold(cl::Buffer& mask, cl::Buffer& mads, cl::Buffer& d_in, float threshold, size_t m, size_t n, size_t local_size_m, size_t local_size_n);

	void MADRows(const cl::Buffer& mads, cl::Buffer& medians, cl::Buffer& d_in, size_t m, size_t n, size_t local_size);

	void ComputeRowMedians(cl::Buffer& medians, cl::Buffer& d_in, size_t m, size_t n, size_t local_size);


	//void Grubb(const cl::Buffer data, size_t len, size_t thread_offset, float threshold, size_t local_size);
	void Grubb(const cl::Buffer data, size_t len, size_t work_per_thread, float threshold, size_t local_size);

};

#endif
