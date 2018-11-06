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
#include <string>
#include <CL/cl.hpp>

#include "timing.h"
#include <opencl_error_handling.h>


class RFIPipeline {
public:
	
	// GPU kernels which this class provides a wrapper for.
	cl::Kernel reduce;
	cl::Kernel mask_rows;
	cl::Kernel transpose;
	cl::Kernel compute_mads;
	cl::Kernel sum_threshold;
	cl::Kernel compute_means;
	cl::Kernel edge_threshold;
	cl::Kernel compute_medians;
	cl::Kernel detect_outliers;
	cl::Kernel compute_deviation;
	cl::Kernel replace_rfi_medians;
	cl::Kernel replace_rfi_constant;
	cl::Kernel mask_row_sum_threshold;

	// OpenCL enviroment variables.
	cl::Program program;
	cl::Context context;
	cl::CommandQueue queue;
	std::vector<cl::Device> devices;	

	// cl_int used to get error reporting from OpenCL.
	cl_int error_code;

	cl::Buffer data_T;
	cl::Buffer mask;
	cl::Buffer mask_T;

	cl::Buffer freq_medians;
	cl::Buffer time_medians;

	cl::Buffer time_mads;
	cl::Buffer freq_mads;

	cl::Buffer time_means;
	cl::Buffer time_temp;

	enum RFIReplaceMode {MEDIANS, ZEROS};

	struct Params {
		int mode;
		int n_iter;
		int n_samples;
		int n_channels;
		int n_padded_samples;
		float mad_threshold;
		float std_threshold;
		RFIReplaceMode rfi_replace_mode;
	};

	const Params params;

	// ********** Class setup functions  ********** // 
	
	RFIPipeline (const Params& params);

	RFIPipeline (cl::Context& context, cl::CommandQueue& queue, 
			     std::vector<cl::Device>& devices, const Params& params);
	
	void LoadKernels (void);

	void InitMemBuffers (const int mode);

	static void ReadConfigFile(const std::string file_name);

	// ********** RFI mitigation pipelines ********** // 
	
	void Flag (const cl::Buffer& data);

	// ********** Memory util functions  ********** // 
	
	void WriteToBuffer (void *host_mem, const cl::Buffer& buffer, const int size) 
	{
		CHECK_CL(queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, size, host_mem));
	}

	void ReadFromBuffer (void *host_mem, const cl::Buffer& buffer, const int size) 
	{
		CHECK_CL(queue.enqueueReadBuffer(buffer, CL_TRUE, 0, size, host_mem));
	}

	void CopyBuffer (const cl::Buffer& src, const cl::Buffer& dest, const int size) 
	{
		CHECK_CL(queue.enqueueCopyBuffer(src, dest, 0, 0, size));
	}

	void ClearBuffer (const cl::Buffer& buffer, const int size) 
	{
		CHECK_CL(queue.enqueueFillBuffer(buffer, 0, 0, size));
	}

	cl::Buffer InitBuffer (const cl_mem_flags mem_flag, const int size) 
	{
		return cl::Buffer(context, mem_flag, size);
	}

	// ********** GPU kernels  ********** // 

	float FloatReduce (const cl::Buffer& d_out, 
			           const cl::Buffer& d_in, 
					   int n);

	void Transpose (const cl::Buffer& d_out, 
			        const cl::Buffer& d_in, 
				    int m, int n, 
				    int nx, int ny);

	void EdgeThreshold (const cl::Buffer& d_out, 
			            const cl::Buffer& d_in, 
					    const cl::Buffer& mads, 
					    float threshold, 
					    int max_window_size, 
					    int m, int n, int N,
					    int nx, int ny);

	void SumThreshold (cl::Buffer& m_out, 
			           const cl::Buffer& d_in, 
					   cl::Buffer& m_in, 
					   const cl::Buffer& thresholds, 
					   int max_window_size, 
					   int m, int n, int N,
					   int nx, int ny);

	void ComputeMads (const cl::Buffer& mads, 
			          const cl::Buffer& medians, 
					  const cl::Buffer& d_in, 
					  int m, int n, int N,
					  int nx, int ny);

	void ComputeMedians (const cl::Buffer& medians, 
			             const cl::Buffer& data, 
						 int m, int n, int N,
						 int nx, int ny);

	void ComputeMeans (const cl::Buffer& d_out, 
			           const cl::Buffer& d_in, 
					   int m, int n, int N);

	float ComputeStd (const cl::Buffer& data, 
			          const cl::Buffer& temp, 
					  float mean, 
					  int n, 
					  int nx);


	void DetectOutliers (const cl::Buffer& d_out, 
			             const cl::Buffer& d_in, 
						 float mean, 
						 float std, 
						 float threshold, 
						 int n, 
						 int nx);

	void MaskRowSumThreshold (const cl::Buffer& m_out, 
							  const cl::Buffer& m_in, 
							  int m, int n, int N,
							  int nx, int ny);


	void MaskRows (const cl::Buffer& m_out, 
			       const cl::Buffer& m_in, 
				   int m, int n, int N,
				   int nx, int ny);

	void ReplaceRFI (const cl::Buffer& d_out, 
			         const cl::Buffer& d_in, 
					 const cl::Buffer& m_in, 
					 const cl::Buffer& new_values, 
					 const RFIReplaceMode& mode,
					 int m, int n, int N,
					 int nx, int ny);
	

};

#endif
