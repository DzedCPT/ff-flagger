/*****************************************************************************
 AUTHOR:
 Jedda Boyle

 CONTAINS:
 The implementation of GPUEnviroment class.

 NOTES:

  *****************************************************************************/

#include <vector>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <cmath>
#include <string>
#include <fstream>
#include <sstream>

#include <CL/cl.hpp>

#include "device.h"
#include "opencl_error_handling.h"

GPUEnviroment::GPUEnviroment (void) {

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

	// Read in file containing the OpenCl code.
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
	mask = cl::Kernel(program, "mask", &error_code); CHECK_CL(error_code);
	upcast = cl::Kernel(program, "upcast", &error_code); CHECK_CL(error_code);
	reduce = cl::Kernel(program, "reduce", &error_code); CHECK_CL(error_code);
	mad_rows = cl::Kernel(program, "mad_rows", &error_code); CHECK_CL(error_code);
	downcast = cl::Kernel(program, "downcast", &error_code); CHECK_CL(error_code);
	transpose = cl::Kernel(program, "transpose", &error_code); CHECK_CL(error_code);
	flag_rows = cl::Kernel(program, "flag_rows", &error_code); CHECK_CL(error_code);
	edge_threshold = cl::Kernel(program, "edge_threshold", &error_code); CHECK_CL(error_code);

}


cl::Buffer GPUEnviroment::InitBuffer(const cl_mem_flags mem_flag, const size_t size) {
	return cl::Buffer(context, mem_flag, size);
}


void GPUEnviroment::WriteToBuffer(void* host_mem, cl::Buffer& buffer, const size_t size) {
	CHECK_CL(queue.enqueueWriteBuffer(buffer, CL_TRUE, 0, size, host_mem));
}


void GPUEnviroment::ReadFromBuffer(void* host_mem, cl::Buffer& buffer, const size_t size) {
	CHECK_CL(queue.enqueueReadBuffer(buffer, CL_TRUE, 0, size, host_mem));
}


void GPUEnviroment::CopyBuffer (const cl::Buffer& src, cl::Buffer& dest, size_t size) {
	CHECK_CL(queue.enqueueCopyBuffer(src, dest, 0, 0, size));
}


void GPUEnviroment::Downcast(const cl::Buffer& d_out, cl::Buffer& d_in, size_t len, size_t local_size) {
	size_t global_size = local_size * std::ceil((float) len / local_size);
	//kernel void upcast(global float *d_out, const global uchar *d_in, uint len) {

	CHECK_CL(downcast.setArg(0, d_out));
	CHECK_CL(downcast.setArg(1, d_in));
	CHECK_CL(downcast.setArg(2, static_cast<unsigned int>(len)));

	CHECK_CL(queue.enqueueNDRangeKernel(downcast, cl::NullRange, global_size, local_size));


//kernel void mask(global float *d_out, const global float *d_in, const global float* mask, uint m, uint n) {

}


void GPUEnviroment::Upcast(const cl::Buffer& d_out, cl::Buffer& d_in, size_t len, size_t local_size) {
	size_t global_size = local_size * std::ceil((float) len / local_size);

	CHECK_CL(upcast.setArg(0, d_out));
	CHECK_CL(upcast.setArg(1, d_in));
	CHECK_CL(upcast.setArg(2, static_cast<unsigned int>(len)));

	CHECK_CL(queue.enqueueNDRangeKernel(upcast, cl::NullRange, global_size, local_size));


}


void GPUEnviroment::Mask(const cl::Buffer& d_out, cl::Buffer& d_in, cl::Buffer& d_mask, float mask_value, size_t m, size_t n, size_t local_size_m, size_t local_size_n) {
	size_t global_size_m = local_size_m * std::ceil((float) m / local_size_m);
	size_t global_size_n = local_size_n * std::ceil((float) n / local_size_n);

	cl::NDRange local_range(local_size_m, local_size_n);
	cl::NDRange global_range(global_size_m, global_size_n);

	CHECK_CL(mask.setArg(0, d_out));
	CHECK_CL(mask.setArg(1, d_in));
	CHECK_CL(mask.setArg(2, d_mask));
	CHECK_CL(mask.setArg(3, mask_value));
	CHECK_CL(mask.setArg(4, static_cast<unsigned int>(m)));
	CHECK_CL(mask.setArg(5, static_cast<unsigned int>(n)));

	CHECK_CL(queue.enqueueNDRangeKernel(mask, cl::NullRange, global_range, local_range));


}


void GPUEnviroment::Transpose(const cl::Buffer& d_out, cl::Buffer& d_in, size_t m, size_t n, size_t local_size_m, size_t local_size_n) {
	size_t global_size_m = local_size_m * std::ceil((float) m / local_size_m);
	size_t global_size_n = local_size_n * std::ceil((float) n / local_size_n);

	cl::NDRange local_range(local_size_m, local_size_n);
	cl::NDRange global_range(global_size_m, global_size_n);

	CHECK_CL(transpose.setArg(0, d_out));
	CHECK_CL(transpose.setArg(1, d_in));
	CHECK_CL(transpose.setArg(2, static_cast<unsigned int>(m)));
	CHECK_CL(transpose.setArg(3, static_cast<unsigned int>(n)));

	CHECK_CL(queue.enqueueNDRangeKernel(transpose, cl::NullRange, global_range, local_range));


}


void GPUEnviroment::EdgeThreshold(cl::Buffer& mask, cl::Buffer& mads, cl::Buffer& d_in, float threshold, size_t m, size_t n, size_t local_size_m, size_t local_size_n) {
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



}

void GPUEnviroment::MADRows(const cl::Buffer& mads, cl::Buffer& medians, cl::Buffer& d_in, size_t m, size_t n, size_t local_size) {
	//size_t global_size_m = local_size_m * std::ceil((float) m / (local_size_m * window_size));
	size_t global_size = local_size * std::ceil((float) m / local_size);

	//void mad_rows(global float *d_out, global float *d_in, uint window_size, uint m, uint n, local float *local_mem) {
	CHECK_CL(mad_rows.setArg(0, mads));
	CHECK_CL(mad_rows.setArg(1, medians));
	CHECK_CL(mad_rows.setArg(2, d_in));
	CHECK_CL(mad_rows.setArg(3, static_cast<unsigned int>(m)));
	CHECK_CL(mad_rows.setArg(4, static_cast<unsigned int>(n)));

	CHECK_CL(queue.enqueueNDRangeKernel(mad_rows, cl::NullRange, global_size, local_size));



}

void GPUEnviroment::FlagRows(const cl::Buffer& mask, float row_sum_threshold, size_t m, size_t n, size_t local_size) {
	size_t global_size = local_size * std::ceil((float) m / local_size);

	CHECK_CL(flag_rows.setArg(0, mask));
	CHECK_CL(flag_rows.setArg(1, row_sum_threshold));
	CHECK_CL(flag_rows.setArg(2, static_cast<unsigned int>(m)));
	CHECK_CL(flag_rows.setArg(3, static_cast<unsigned int>(n)));

	CHECK_CL(queue.enqueueNDRangeKernel(flag_rows, cl::NullRange, global_size, local_size));



}


float GPUEnviroment::Reduce(const cl::Buffer d_in, size_t drop_out, size_t len, size_t local_size) {
	// loCHECKcal_size cannot equal 1.
	local_size = std::max(local_size, (size_t) 2);
	size_t num_groups = std::ceil((float) len / local_size);
	size_t global_size = local_size * num_groups;

	// Setup device mem required for reduction.
	cl::Buffer buffer_A = InitBuffer(CL_MEM_READ_WRITE, len * sizeof(float));
	cl::Buffer buffer_B = InitBuffer(CL_MEM_READ_WRITE, num_groups * sizeof(float));
	
	// Init buffer_A to have the same values as d_in.
	CopyBuffer (d_in, buffer_A, len * sizeof(float));

	while (true) {
		// Set args.
		CHECK_CL(reduce.setArg(0, buffer_A));
		CHECK_CL(reduce.setArg(1, buffer_B));
		CHECK_CL(reduce.setArg(2, static_cast<unsigned int>(len)));
		CHECK_CL(reduce.setArg(3, local_size * sizeof(float), NULL));

		// Launch kernel.
		CHECK_CL(queue.enqueueNDRangeKernel(reduce, cl::NullRange, global_size, local_size));

		std::swap(buffer_A, buffer_B);

		// Check if the number of values still to be reduced is less than drop out.
		if (num_groups < drop_out) {
			break;
		}

		// Set values for next reduce step on partally reduced values.
		len = num_groups;	
		num_groups = std::ceil((float) len / local_size);
		global_size = local_size * num_groups;
		
	}

	// Perform the final reduction sequentially.
	std::vector<float> partially_reduced(num_groups);
	ReadFromBuffer(partially_reduced.data(), buffer_A,  num_groups * sizeof(float));
	return std::accumulate(partially_reduced.begin(), partially_reduced.end(), 0.0);
}


//void GPUEnviroment::MADRows(const cl::Buffer& d_out, cl::Buffer& d_in, size_t mad_size, size_t m, size_t n, size_t local_size_m, size_t local_size_n) {
	////size_t global_size_m = local_size_m * std::ceil((float) m / (local_size_m * window_size));
	//size_t global_size_m = local_size_m * std::ceil((float) m / local_size_m);
	//size_t global_size_n = local_size_n * std::ceil((float) n / (local_size_n * mad_size));

	//cl::NDRange local_range(local_size_m, local_size_n);
	//cl::NDRange global_range(global_size_m, global_size_n);

	////void mad_rows(global float *d_out, global float *d_in, uint window_size, uint m, uint n, local float *local_mem) {
	//CHECK_CL(mad_rows.setArg(0, d_out));
	//CHECK_CL(mad_rows.setArg(1, d_in));
	//CHECK_CL(mad_rows.setArg(2, static_cast<unsigned int>(mad_size)));
	//CHECK_CL(mad_rows.setArg(3, static_cast<unsigned int>(m)));
	//CHECK_CL(mad_rows.setArg(4, static_cast<unsigned int>(n)));
	//CHECK_CL(mad_rows.setArg(5, local_size_m * local_size_n * 256 * sizeof(float), NULL));

	//CHECK_CL(queue.enqueueNDRangeKernel(mad_rows, cl::NullRange, global_range, local_range));



//}

