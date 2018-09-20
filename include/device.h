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

class GPUEnviroment {
public:
	
	// GPU kernels which this class provides a wrapper for.
	cl::Kernel mask;
	cl::Kernel upcast;
	cl::Kernel downcast;
	cl::Kernel transpose;
	cl::Kernel mad_rows;
	cl::Kernel edge_threshold;
	cl::Kernel flag_rows;
	cl::Kernel reduce;

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

	void WriteToBuffer(void* host_mem, cl::Buffer& buffer, const size_t size);

	void ReadFromBuffer(void* host_mem, cl::Buffer& buffer, const size_t size);

	void CopyBuffer (const cl::Buffer& src, cl::Buffer& dest, size_t size);

	void Mask(const cl::Buffer& d_out, cl::Buffer& d_in, cl::Buffer& d_mask, float mask_value, size_t m, size_t n, size_t local_size_m, size_t local_size_n);

	void Upcast(const cl::Buffer& d_out, cl::Buffer& d_in, size_t len, size_t local_size);

	void Downcast(const cl::Buffer& d_out, cl::Buffer& d_in, size_t len, size_t local_size);

	void Transpose(const cl::Buffer& d_out, cl::Buffer& d_in, size_t m, size_t n, size_t local_size_m, size_t local_size_n);

	void EdgeThreshold(cl::Buffer& mask, cl::Buffer& mads, cl::Buffer& d_in, float threshold, size_t m, size_t n, size_t local_size_m, size_t local_size_n);

	void MADRows(const cl::Buffer& mads, cl::Buffer& medians, cl::Buffer& d_in, size_t m, size_t n, size_t local_size);

	void FlagRows(const cl::Buffer& mask, float row_sum_threshold, size_t m, size_t n, size_t local_size);

	float Reduce(const cl::Buffer d_in, size_t drop_out, size_t len, size_t local_size);

};

#endif
