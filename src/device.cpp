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
#include <ctime>
#include <string>
#include <fstream>
#include <sstream>
#include <cassert>

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
	mask_rows = cl::Kernel(program, "mask_rows", &error_code); CHECK_CL(error_code);
	constant_row_mask = cl::Kernel(program, "constant_row_mask", &error_code); CHECK_CL(error_code);
	grubb = cl::Kernel(program, "grubb", &error_code); CHECK_CL(error_code);
	mad_rows = cl::Kernel(program, "mad_rows", &error_code); CHECK_CL(error_code);
	transpose = cl::Kernel(program, "transpose", &error_code); CHECK_CL(error_code);
	edge_threshold = cl::Kernel(program, "edge_threshold", &error_code); CHECK_CL(error_code);
	row_medians = cl::Kernel(program, "row_medians", &error_code); CHECK_CL(error_code);

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

void GPUEnviroment::MaskRows(const cl::Buffer& data, cl::Buffer& mask, cl::Buffer& medians, size_t m, size_t n, size_t local_size) {
	MARK_TIME(mark);
	size_t global_size = local_size * std::ceil((float) m / local_size);

	CHECK_CL(mask_rows.setArg(0, data));
	CHECK_CL(mask_rows.setArg(1, mask));
	CHECK_CL(mask_rows.setArg(2, medians));
	CHECK_CL(mask_rows.setArg(3, static_cast<unsigned int>(m)));
	CHECK_CL(mask_rows.setArg(4, static_cast<unsigned int>(n)));

	CHECK_CL(queue.enqueueNDRangeKernel(mask_rows, cl::NullRange, global_size, local_size));

	ADD_TIME_SINCE_MARK(mask_rows_timer, mark);

}

void GPUEnviroment::ConstantRowMask(const cl::Buffer& data, cl::Buffer& mask, size_t m, size_t n, size_t local_size) {
	MARK_TIME(mark);
	size_t global_size = local_size * std::ceil((float) m / local_size);

	CHECK_CL(constant_row_mask.setArg(0, data));
	CHECK_CL(constant_row_mask.setArg(1, mask));
	CHECK_CL(constant_row_mask.setArg(2, static_cast<unsigned int>(m)));
	CHECK_CL(constant_row_mask.setArg(3, static_cast<unsigned int>(n)));

	CHECK_CL(queue.enqueueNDRangeKernel(constant_row_mask, cl::NullRange, global_size, local_size));

	ADD_TIME_SINCE_MARK(const_mask_rows_timer, mark);

}


//void GPUEnviroment::Mask(const cl::Buffer& d_out, cl::Buffer& d_in, cl::Buffer& d_mask, uint8_t mask_value, size_t m, size_t n, size_t local_size_m, size_t local_size_n) {
void GPUEnviroment::Mask(const cl::Buffer& d_out, cl::Buffer& d_in, cl::Buffer& d_mask, cl::Buffer& freq_medians, size_t m, size_t n, size_t local_size_m, size_t local_size_n) {

	MARK_TIME(mark);

	size_t global_size_m = local_size_m * std::ceil((float) m / local_size_m);
	size_t global_size_n = local_size_n * std::ceil((float) n / local_size_n);

	cl::NDRange local_range(local_size_m, local_size_n);
	cl::NDRange global_range(global_size_m, global_size_n);

	CHECK_CL(mask.setArg(0, d_out));
	CHECK_CL(mask.setArg(1, d_in));
	CHECK_CL(mask.setArg(2, d_mask));
	CHECK_CL(mask.setArg(3, freq_medians));
	CHECK_CL(mask.setArg(4, static_cast<unsigned int>(m)));
	CHECK_CL(mask.setArg(5, static_cast<unsigned int>(n)));

	CHECK_CL(queue.enqueueNDRangeKernel(mask, cl::NullRange, global_range, local_range));

	ADD_TIME_SINCE_MARK(mask_timer, mark);


}

void GPUEnviroment::Transpose( cl::Buffer& d_out, cl::Buffer& d_in, size_t m, size_t n, size_t local_size_m, size_t local_size_n) {
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


//void GPUEnviroment::Transpose(cl::Buffer& d_out, cl::Buffer& d_in, size_t m, size_t n, size_t tile_dim, size_t local_size_m) {
	
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


void GPUEnviroment::EdgeThreshold(cl::Buffer& mask, cl::Buffer& mads, cl::Buffer& d_in, float threshold, size_t m, size_t n, size_t local_size_m, size_t local_size_n) {
	
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

void GPUEnviroment::ComputeRowMedians(cl::Buffer& medians, cl::Buffer& d_in, size_t m, size_t n, size_t local_size) {
	MARK_TIME(mark);

	size_t global_size = local_size * std::ceil((float) m / local_size);

	CHECK_CL(row_medians.setArg(0, medians));
	CHECK_CL(row_medians.setArg(1, d_in));
	CHECK_CL(row_medians.setArg(2, static_cast<unsigned int>(m)));
	CHECK_CL(row_medians.setArg(3, static_cast<unsigned int>(n)));

	CHECK_CL(queue.enqueueNDRangeKernel(row_medians, cl::NullRange, global_size, local_size));

	ADD_TIME_SINCE_MARK(medians_timer, mark);

}


void GPUEnviroment::MADRows(const cl::Buffer& mads, cl::Buffer& medians, cl::Buffer& d_in, size_t m, size_t n, size_t local_size) {
	MARK_TIME(mark);
	//size_t global_size_m = local_size_m * std::ceil((float) m / (local_size_m * window_size));
	size_t global_size = local_size * std::ceil((float) m / local_size);

	//void mad_rows(global float *d_out, global float *d_in, uint window_size, uint m, uint n, local float *local_mem) {
	CHECK_CL(mad_rows.setArg(0, mads));
	CHECK_CL(mad_rows.setArg(1, medians));
	CHECK_CL(mad_rows.setArg(2, d_in));
	CHECK_CL(mad_rows.setArg(3, static_cast<unsigned int>(m)));
	CHECK_CL(mad_rows.setArg(4, static_cast<unsigned int>(n)));

	CHECK_CL(queue.enqueueNDRangeKernel(mad_rows, cl::NullRange, global_size, local_size));

	ADD_TIME_SINCE_MARK(mad_timer, mark);


}

//void GPUEnviroment::Grubb(const cl::Buffer data, size_t len, size_t size, float threshold, size_t local_size) {
void GPUEnviroment::Grubb(const cl::Buffer data, size_t len, size_t work_per_thread, float threshold, size_t local_size) {
	MARK_TIME(mark);
	
	size_t global_size = local_size * std::ceil((float) len / (local_size * work_per_thread));

	//grubb(global uchar *data, uint len, uint work_per_thread, float threshold, local float *local_mem) {

	CHECK_CL(grubb.setArg(0, data));
	CHECK_CL(grubb.setArg(1, static_cast<unsigned int>(len)));
	CHECK_CL(grubb.setArg(2, static_cast<unsigned int>(work_per_thread)));
	//CHECK_CL(grubb.setArg(2, threshold));
	CHECK_CL(grubb.setArg(3, threshold));
	CHECK_CL(grubb.setArg(4, local_size * work_per_thread * sizeof(float), NULL));
	CHECK_CL(grubb.setArg(5, local_size * sizeof(float), NULL));

	CHECK_CL(queue.enqueueNDRangeKernel(grubb, cl::NullRange, global_size, local_size));
	ADD_TIME_SINCE_MARK(grubb_timer, mark);

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

