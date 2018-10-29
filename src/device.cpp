/*****************************************************************************
 AUTHOR:
 Jedda Boyle

 CONTAINS:
 The implementation of RFIPipeline class.

 NOTES:

  *****************************************************************************/

#include <vector>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <cmath>
#include <ctime>
#include <string>
#include <fstream>
#include <sstream>
#include <cassert>

#include <CL/cl.hpp>

#include "device.h"
#include "opencl_error_handling.h"

void RFIPipeline::LoadKernels(void) {
	std::ifstream file_stream("src/kernels.cl");
	std::stringstream buffer;
	buffer << file_stream.rdbuf();	
	std::string source = buffer.str();
	const char * c = source.c_str();

	// Create program.
	cl::Program::Sources program_source(1, std::make_pair(c, strlen(c)));
	program = cl::Program(context, program_source);
	CHECK_CL(program.build(devices));
	
	// Create kernels.
	replace_rfi = cl::Kernel(program, "replace_rfi", &error_code); CHECK_CL(error_code);
	mask_rows = cl::Kernel(program, "mask_rows", &error_code); CHECK_CL(error_code);
	constant_row_mask = cl::Kernel(program, "constant_row_mask", &error_code); CHECK_CL(error_code);
	detect_outliers = cl::Kernel(program, "detect_outliers", &error_code); CHECK_CL(error_code);
	compute_mads = cl::Kernel(program, "compute_mads", &error_code); CHECK_CL(error_code);
	transpose = cl::Kernel(program, "transpose", &error_code); CHECK_CL(error_code);
	edge_threshold = cl::Kernel(program, "edge_threshold", &error_code); CHECK_CL(error_code);
	compute_medians = cl::Kernel(program, "compute_medians", &error_code); CHECK_CL(error_code);
	reduce = cl::Kernel(program, "reduce", &error_code); CHECK_CL(error_code);
	compute_means = cl::Kernel(program, "compute_means", &error_code); CHECK_CL(error_code);
	compute_deviation = cl::Kernel(program, "compute_deviation", &error_code); CHECK_CL(error_code);


}


void RFIPipeline::InitMemBuffers(void) {
	data_T = this->InitBuffer(CL_MEM_READ_WRITE, n_channels * n_samples * sizeof(uint8_t));
	mask   = this->InitBuffer(CL_MEM_READ_WRITE, n_channels * n_samples * sizeof(uint8_t));
	mask_T = this->InitBuffer(CL_MEM_READ_WRITE, n_channels * n_samples * sizeof(uint8_t));

	time_mads = this->InitBuffer(CL_MEM_READ_WRITE, n_samples * sizeof(uint8_t));
	time_medians = this->InitBuffer(CL_MEM_READ_WRITE, n_samples * sizeof(uint8_t));
	freq_mads = this->InitBuffer(CL_MEM_READ_WRITE, n_channels * sizeof(uint8_t));
	freq_medians = this->InitBuffer(CL_MEM_READ_WRITE, n_channels * sizeof(uint8_t));

	
	partially_reduced.resize(100);


}


void RFIPipeline::AAFlagger(const cl::Buffer& data) {

	//queue.enqueueFillBuffer(mask, 0, 0, n_channels * n_samples * sizeof(uint8_t));

	//ComputeMads(freq_mads, freq_medians, data, n_channels, n_samples, 500);
	//EdgeThreshold(mask, freq_mads, data, 3.5, n_channels, n_samples, 12, 12);

	//Transpose(data_T, data, n_channels, n_samples, 12, 12);
	//Transpose(mask_T, mask, n_channels, n_samples, 12, 12);
	

	//ComputeMads(time_mads, time_medians, data_T, n_samples, n_channels, 500);
	//EdgeThreshold(mask_T, time_mads, data_T, 3.5, n_samples, n_channels, 12, 12);
	//OutlierDetection(time_medians, n_samples, 5, 3.5, 1000);
	//ConstantRowMask(mask_T, time_medians, n_samples, n_channels, 500);

	//Transpose(data, data_T, n_samples, n_channels, 12, 12);
	//Transpose(mask, mask_T, n_samples, n_channels, 12, 12);

	//ReplaceRFI(data, data, mask, freq_medians, n_channels, n_samples, 12, 12);

	//this->queue.enqueueFillBuffer(mask, 0, 0, n * m * sizeof(uint8_t));

	//ComputeMads(time_mads, time_medians, uint_buffer, m, n, 500);
	//EdgeThreshold(mask, time_mads, uint_buffer, threshold, m, n, 12, 12);
	//OutlierDetection(time_medians, m, 5, threshold, 1000);
	//ConstantRowMask(mask, time_medians, m, n, 500);

	//gpu.Transpose(uint_buffer_T, uint_buffer, m, n, 12, 12);
	//gpu.Transpose(mask_T, mask, m, n, 12, 12);

	//gpu.ComputeMads(freq_mads, freq_medians, uint_buffer_T, n, m, 500);
	//gpu.EdgeThreshold(mask_T, freq_mads, uint_buffer_T, threshold, n, m, 12, 12);

	//gpu.ReplaceRFI(uint_buffer_T, uint_buffer_T, mask_T, freq_medians, m, n, 12, 12);

	//gpu.Transpose(uint_buffer, uint_buffer_T, n, m, 12, 12);
	
}

RFIPipeline::RFIPipeline (cl::Context& context, cl::CommandQueue& queue, std::vector<cl::Device>& devices, size_t _n_channels, size_t _n_samples) {
	n_samples = _n_samples;
	n_channels = _n_channels;

	this->devices = devices;
	this->queue = queue;
	this->context = context;

	this->LoadKernels();

	this->InitMemBuffers();

}


RFIPipeline::RFIPipeline (size_t _n_channels, size_t _n_samples) {
	n_samples = _n_samples;
	n_channels = _n_channels;


	// Get platform.
	std::vector<cl::Platform> platforms;
	CHECK_CL(cl::Platform::get(&platforms));

	if (platforms.empty()) { 
		std::cerr << "OpenCL platforms not found." << std::endl;
		exit(1);
	}

	// Get first available GPU device.
	for (auto platform = platforms.begin(); devices.empty() && platform != platforms.end(); platform++) {
		std::vector<cl::Device> platform_devices;

		platform->getDevices(CL_DEVICE_TYPE_GPU, &platform_devices);

		for(auto device = platform_devices.begin(); devices.empty() && device != platform_devices.end(); device++) {
			if (!device->getInfo<CL_DEVICE_AVAILABLE>()) continue;

			devices.push_back(*device);
			context = cl::Context(devices);
		}
	}

	if (devices.empty()) {
		std::cerr << "No available GPUs found." << std::endl;
		exit(1);
	}

	// Create command queue.
	queue = cl::CommandQueue(context, devices[0], 0, &error_code); CHECK_CL(error_code);

	this->LoadKernels();
	this->InitMemBuffers();
}

//RFIPipeline::~RFIPipeline (void) {
	//clReleaseMemObject(data_T);
	//clReleaseMemObject(mask);
	//clReleaseMemObject(mask_T);

	//clReleaseMemObject(time_mads);
	//clReleaseMemObject(time_medians);
	//clReleaseMemObject(freq_mads);
	//clReleaseMemObject(freq_medians);
//}


cl::Buffer RFIPipeline::InitBuffer(const cl_mem_flags mem_flag, const size_t size) {
	return cl::Buffer(context, mem_flag, size);
}



void RFIPipeline::WriteToBuffer(void* host_mem, cl::Buffer& buffer, const size_t size) {
	CHECK_CL(queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, size, host_mem));
}


void RFIPipeline::ReadFromBuffer(void* host_mem, cl::Buffer& buffer, const size_t size) {
	CHECK_CL(queue.enqueueReadBuffer(buffer, CL_TRUE, 0, size, host_mem));
}


void RFIPipeline::CopyBuffer (const cl::Buffer& src, cl::Buffer& dest, size_t size) {
	CHECK_CL(queue.enqueueCopyBuffer(src, dest, 0, 0, size));
}

//void RFIPipeline::MaskRows(const cl::Buffer& data, cl::Buffer& mask, cl::Buffer& medians, size_t m, size_t n, size_t local_size) {
void RFIPipeline::MaskRows(const cl::Buffer& d_out, cl::Buffer& mask, size_t m, size_t n, size_t local_size_m, size_t local_size_n) {
	MARK_TIME(mark);
	size_t global_size_m = local_size_m * std::ceil((float) m / local_size_m);

	cl::NDRange local_range(local_size_m, local_size_n);
	cl::NDRange global_range(global_size_m, local_size_n);

	CHECK_CL(mask_rows.setArg(0, d_out));
	CHECK_CL(mask_rows.setArg(1, mask));
	CHECK_CL(mask_rows.setArg(2, static_cast<unsigned int>(m)));
	CHECK_CL(mask_rows.setArg(3, static_cast<unsigned int>(n)));

	CHECK_CL(queue.enqueueNDRangeKernel(mask_rows, cl::NullRange, global_range, local_range));

	ADD_TIME_SINCE_MARK(mask_rows_timer, mark);

}

void RFIPipeline::ConstantRowMask(const cl::Buffer& data, cl::Buffer& mask, size_t m, size_t n, size_t local_size) {
	MARK_TIME(mark);
	size_t global_size = local_size * std::ceil((float) m / local_size);

	CHECK_CL(constant_row_mask.setArg(0, data));
	CHECK_CL(constant_row_mask.setArg(1, mask));
	CHECK_CL(constant_row_mask.setArg(2, static_cast<unsigned int>(m)));
	CHECK_CL(constant_row_mask.setArg(3, static_cast<unsigned int>(n)));

	CHECK_CL(queue.enqueueNDRangeKernel(constant_row_mask, cl::NullRange, global_size, local_size));

	ADD_TIME_SINCE_MARK(const_mask_rows_timer, mark);

}


void RFIPipeline::ReplaceRFI(const cl::Buffer& d_out, const cl::Buffer& d_in, const cl::Buffer& d_mask, const cl::Buffer& freq_medians, size_t m, size_t n, size_t local_size_m, size_t local_size_n) {

	MARK_TIME(mark);

	size_t global_size_m = local_size_m * std::ceil((float) m / local_size_m);
	size_t global_size_n = local_size_n * std::ceil((float) n / local_size_n);

	cl::NDRange local_range(local_size_m, local_size_n);
	cl::NDRange global_range(global_size_m, global_size_n);

	CHECK_CL(replace_rfi.setArg(0, d_out));
	CHECK_CL(replace_rfi.setArg(1, d_in));
	CHECK_CL(replace_rfi.setArg(2, d_mask));
	CHECK_CL(replace_rfi.setArg(3, freq_medians));
	CHECK_CL(replace_rfi.setArg(4, static_cast<unsigned int>(m)));
	CHECK_CL(replace_rfi.setArg(5, static_cast<unsigned int>(n)));

	CHECK_CL(queue.enqueueNDRangeKernel(replace_rfi, cl::NullRange, global_range, local_range));

	ADD_TIME_SINCE_MARK(replace_rfi_timer, mark);


}

void RFIPipeline::Transpose(const cl::Buffer& d_out, const cl::Buffer& d_in, size_t m, size_t n, size_t local_size_m, size_t local_size_n) {
	MARK_TIME(mark);
	size_t global_size_m = local_size_m * std::ceil((float) m / local_size_m);
	size_t global_size_n = local_size_n * std::ceil((float) n / local_size_n);
	

	cl::NDRange local_range(local_size_m, local_size_n);
	cl::NDRange global_range(global_size_m, global_size_n);

	CHECK_CL(transpose.setArg(0, d_out));
	CHECK_CL(transpose.setArg(1, d_in));
	CHECK_CL(transpose.setArg(2, static_cast<unsigned int>(m)));
	CHECK_CL(transpose.setArg(3, static_cast<unsigned int>(n)));

	CHECK_CL(queue.enqueueNDRangeKernel(transpose, cl::NullRange, global_range, local_range));


	ADD_TIME_SINCE_MARK(transpose_timer, mark);
}


//void RFIPipeline::Transpose(cl::Buffer& d_out, cl::Buffer& d_in, size_t m, size_t n, size_t tile_dim, size_t local_size_m) {
	
	//MARK_TIME(mark);
	//// Must be square.
	//size_t local_size_n = tile_dim;
	//assert(tile_dim % local_size_m == 0);

	//size_t global_size_m = local_size_m * std::ceil((float) m / tile_dim);
	//size_t global_size_n = local_size_n * std::ceil((float) n / local_size_n);

	//cl::NDRange local_range(local_size_m, local_size_n);
	//cl::NDRange global_range(global_size_m, global_size_n);


	//CHECK_CL(transpose.setArg(0, d_out));
	//CHECK_CL(transpose.setArg(1, d_in));
	//CHECK_CL(transpose.setArg(2, static_cast<unsigned int>(tile_dim)));
	////CHECK_CL(transpose.setArg(3, static_cast<unsigned int>(tile_dim_n)));
	//CHECK_CL(transpose.setArg(3, static_cast<unsigned int>(m)));
	//CHECK_CL(transpose.setArg(4, static_cast<unsigned int>(n)));
	//CHECK_CL(transpose.setArg(5, tile_dim * tile_dim * sizeof(uint8_t), NULL));

	//CHECK_CL(queue.enqueueNDRangeKernel(transpose, cl::NullRange, global_range, local_range));

	//ADD_TIME_SINCE_MARK(transpose_timer, mark);

//}


void RFIPipeline::EdgeThreshold(const cl::Buffer& mask, const cl::Buffer& mads, const cl::Buffer& d_in, float threshold, size_t m, size_t n, size_t local_size_m, size_t local_size_n) {
	
	MARK_TIME(mark);
	size_t global_size_m = local_size_m * std::ceil((float) m / local_size_m);
	size_t global_size_n = local_size_n * std::ceil((float) n / local_size_n);

	cl::NDRange local_range(local_size_m, local_size_n);
	cl::NDRange global_range(global_size_m, global_size_n);

	CHECK_CL(edge_threshold.setArg(0, mask));
	CHECK_CL(edge_threshold.setArg(1, mads));
	CHECK_CL(edge_threshold.setArg(2, d_in));
	CHECK_CL(edge_threshold.setArg(3, threshold));
	CHECK_CL(edge_threshold.setArg(4, static_cast<unsigned int>(m)));
	CHECK_CL(edge_threshold.setArg(5, static_cast<unsigned int>(n)));

	CHECK_CL(queue.enqueueNDRangeKernel(edge_threshold, cl::NullRange, global_range, local_range));
	ADD_TIME_SINCE_MARK(edge_timer, mark);



}

void RFIPipeline::ComputeMedians(const cl::Buffer& medians, const cl::Buffer& data, size_t m, size_t n, size_t local_size) {
	MARK_TIME(mark);

	size_t global_size = local_size * std::ceil((float) m / local_size);

	CHECK_CL(compute_medians.setArg(0, medians));
	CHECK_CL(compute_medians.setArg(1, data));
	CHECK_CL(compute_medians.setArg(2, static_cast<unsigned int>(m)));
	CHECK_CL(compute_medians.setArg(3, static_cast<unsigned int>(n)));

	CHECK_CL(queue.enqueueNDRangeKernel(compute_medians, cl::NullRange, global_size, local_size));

	ADD_TIME_SINCE_MARK(medians_timer, mark);

}


void RFIPipeline::ComputeMads(const cl::Buffer& mads, const cl::Buffer& medians, const cl::Buffer& d_in, size_t m, size_t n, size_t local_size) {
	MARK_TIME(mark);
	//size_t global_size_m = local_size_m * std::ceil((float) m / (local_size_m * window_size));
	size_t global_size = local_size * std::ceil((float) m / local_size);

	//void compute_mads(global float *d_out, global float *d_in, uint window_size, uint m, uint n, local float *local_mem) {
	CHECK_CL(compute_mads.setArg(0, mads));
	CHECK_CL(compute_mads.setArg(1, medians));
	CHECK_CL(compute_mads.setArg(2, d_in));
	CHECK_CL(compute_mads.setArg(3, static_cast<unsigned int>(m)));
	CHECK_CL(compute_mads.setArg(4, static_cast<unsigned int>(n)));

	CHECK_CL(queue.enqueueNDRangeKernel(compute_mads, cl::NullRange, global_size, local_size));

	ADD_TIME_SINCE_MARK(mad_timer, mark);


}

//void RFIPipeline::Grubb(const cl::Buffer data, size_t len, size_t size, float threshold, size_t local_size) {
void RFIPipeline::DetectOutliers(const cl::Buffer& d_out, const cl::Buffer& d_in, float mean, float std, float threshold, size_t len, size_t local_size) {
	size_t global_size = local_size * std::ceil((float) len / local_size);
//void detect_outliers(global float *d_out, global float *d_in, float mean, float std, float threshold, uint len) { 
	CHECK_CL(detect_outliers.setArg(0, d_out));
	CHECK_CL(detect_outliers.setArg(1, d_in));
	CHECK_CL(detect_outliers.setArg(2, mean));
	CHECK_CL(detect_outliers.setArg(3, std));
	CHECK_CL(detect_outliers.setArg(4, threshold));
	CHECK_CL(detect_outliers.setArg(5, static_cast<unsigned int>(len)));

	CHECK_CL(queue.enqueueNDRangeKernel(detect_outliers, cl::NullRange, global_size, local_size));


}
//void RFIPipeline::OutlierDetection(const cl::Buffer data, size_t len, size_t work_per_thread, float threshold, size_t local_size) {
	//MARK_TIME(mark);
	
	//size_t global_size = local_size * std::ceil((float) len / (local_size * work_per_thread));

	////detect_outliers(global uchar *data, uint len, uint work_per_thread, float threshold, local float *local_mem) {

	//CHECK_CL(detect_outliers.setArg(0, data));
	//CHECK_CL(detect_outliers.setArg(1, static_cast<unsigned int>(len)));
	//CHECK_CL(detect_outliers.setArg(2, static_cast<unsigned int>(work_per_thread)));
	//CHECK_CL(detect_outliers.setArg(3, threshold));
	//CHECK_CL(detect_outliers.setArg(4, local_size * work_per_thread * sizeof(float), NULL));
	//CHECK_CL(detect_outliers.setArg(5, local_size * sizeof(float), NULL));

	//CHECK_CL(queue.enqueueNDRangeKernel(detect_outliers, cl::NullRange, global_size, local_size));
	//ADD_TIME_SINCE_MARK(detect_outliers_timer, mark);

//}

void RFIPipeline::FloatReduce(cl::Buffer& d_out, cl::Buffer& d_in, size_t len, size_t local_size) {
	MARK_TIME(mark);
	size_t num_groups = std::ceil((float) len / local_size);
	size_t global_size = local_size * num_groups;

	// Init buffer_A to have the same values as d_in.
	CHECK_CL(reduce.setArg(0, d_in));
	CHECK_CL(reduce.setArg(1, d_out));
	CHECK_CL(reduce.setArg(2, static_cast<unsigned int>(len)));
	CHECK_CL(reduce.setArg(3, local_size * sizeof(float), NULL));

	CHECK_CL(queue.enqueueNDRangeKernel(reduce, cl::NullRange, global_size, local_size));
	len = num_groups;	
	num_groups = std::ceil((float) len / local_size);
	global_size = local_size * num_groups;
		

	while (true) {
		// Set args.
		CHECK_CL(reduce.setArg(0, d_out));
		//CHECK_CL(reduce.setArg(0, buffer_A));
	  	CHECK_CL(reduce.setArg(1, d_out));
	  	//CHECK_CL(reduce.setArg(1, buffer_A));
	  	CHECK_CL(reduce.setArg(2, static_cast<unsigned int>(len)));
	  	CHECK_CL(reduce.setArg(3, local_size * sizeof(float), NULL));

		// Launch kernel.
	  	CHECK_CL(queue.enqueueNDRangeKernel(reduce, cl::NullRange, global_size, local_size));
	
		// Check if the number of values still to be reduced is less than drop out.
		if (num_groups < 2) {
			break;
		}
		
		// Set values for next reduce step on partially reduced values.
		len = num_groups;	
		num_groups = std::ceil((float) len / local_size);
		global_size = local_size * num_groups;
		//if (num_groups < 2) {
			//break;
		//}
	  
		
	}

	// Perform the final reduction sequentially.
	//std::vector<float> partially_reduced(num_groups);
	//partially_reduced.resize(num_groups);
	//float to_return;
	//ReadFromBuffer(&to_return, d_out,  sizeof(float));
	//ReadFromBuffer(partially_reduced.data(), buffer_A,  num_groups * sizeof(float));
	ADD_TIME_SINCE_MARK(reduce_timer, mark);
	//return std::accumulate(partially_reduced.begin(), partially_reduced.begin() + num_groups, 0.0);
	//return to_return;

}

float RFIPipeline::ComputeStd(cl::Buffer& data, cl::Buffer& temp, float mean, size_t len, size_t local_size) {
	size_t global_size = local_size * std::ceil((float) len / local_size);

	CHECK_CL(compute_deviation.setArg(0, temp));
	CHECK_CL(compute_deviation.setArg(1, data));
	CHECK_CL(compute_deviation.setArg(2, mean));
	CHECK_CL(compute_deviation.setArg(3, static_cast<unsigned int>(len)));

	CHECK_CL(queue.enqueueNDRangeKernel(compute_deviation, cl::NullRange, global_size, local_size));

	//return std::sqrt(FloatReduce(temp, len, local_size));

	

}



void RFIPipeline::ComputeMeans(const cl::Buffer& d_out, const cl::Buffer& d_in, size_t m, size_t n, size_t local_size_m, size_t local_size_n) {
	MARK_TIME(mark);
	size_t global_size_m = local_size_m * std::ceil((float) m / local_size_m);

	cl::NDRange local_range(local_size_m, local_size_n);
	cl::NDRange global_range(global_size_m, local_size_n);


	// Set args.
	CHECK_CL(compute_means.setArg(0, d_out));
	CHECK_CL(compute_means.setArg(1, d_in));
	CHECK_CL(compute_means.setArg(2, static_cast<unsigned int>(m)));
	CHECK_CL(compute_means.setArg(3, static_cast<unsigned int>(n)));
	CHECK_CL(compute_means.setArg(4, local_size_m * local_size_n * sizeof(int), NULL));
					
	// Run kernel.
	CHECK_CL(queue.enqueueNDRangeKernel(compute_means, cl::NullRange, global_range, local_range));

	ADD_TIME_SINCE_MARK(row_mean_timer, mark);

}
//void RFIPipeline::MADRows(const cl::Buffer& d_out, cl::Buffer& d_in, size_t mad_size, size_t m, size_t n, size_t local_size_m, size_t local_size_n) {
	////size_t global_size_m = local_size_m * std::ceil((float) m / (local_size_m * window_size));
	//size_t global_size_m = local_size_m * std::ceil((float) m / local_size_m);
	//size_t global_size_n = local_size_n * std::ceil((float) n / (local_size_n * mad_size));

	//cl::NDRange local_range(local_size_m, local_size_n);
	//cl::NDRange global_range(global_size_m, global_size_n);

	////void compute_mads(global float *d_out, global float *d_in, uint window_size, uint m, uint n, local float *local_mem) {
	//CHECK_CL(compute_mads.setArg(0, d_out));
	//CHECK_CL(compute_mads.setArg(1, d_in));
	//CHECK_CL(compute_mads.setArg(2, static_cast<unsigned int>(mad_size)));
	//CHECK_CL(compute_mads.setArg(3, static_cast<unsigned int>(m)));
	//CHECK_CL(compute_mads.setArg(4, static_cast<unsigned int>(n)));
	//CHECK_CL(compute_mads.setArg(5, local_size_m * local_size_n * 256 * sizeof(float), NULL));

	//CHECK_CL(queue.enqueueNDRangeKernel(compute_mads, cl::NullRange, global_range, local_range));



//}

