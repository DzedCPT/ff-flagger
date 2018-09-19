#ifndef KERNELS_H
#define KERNELS_H

#include <vector>
#include <cstddef>
#include <numeric> 
#include <algorithm>
#include <iostream>
#include <iterator>
#include <cmath>
#include <typeinfo>
#include <functional>

#include <string>
#include <fstream>
#include <sstream>

#include <CL/cl.hpp>

#include "event.h"
#include "opencl_error_handling.h"

class BandSelectorKernel {
public:
	cl::Kernel kernel;

	cl::Program program;

	cl::Context context;

	std::vector<cl::Device> devices;	

	cl::CommandQueue queue;

	cl::Buffer d_data;

	cl_int error_code;

	const size_t m;
	const size_t n;
	const size_t local_work_group_size = 500;
	const float threshold;

	BandSelectorKernel (const size_t _m, const size_t _n, float _threshold): m(_m), n(_n), threshold(_threshold) {

		// Get platform.
		std::vector<cl::Platform> platforms;
		error_code = cl::Platform::get(&platforms);
		CHECK_CL(error_code);

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
		queue = cl::CommandQueue(context, devices[0], 0, &error_code);
		CHECK_CL(error_code);

		// Read in file containing the OpenCl code.
		std::ifstream file_stream("src/kernels.cl");
		std::stringstream buffer;
		buffer << file_stream.rdbuf();	
		std::string source = buffer.str();
		const char * c = source.c_str();

		// Create program.
		cl::Program::Sources program_source(1, std::make_pair(c, strlen(c)));
		program = cl::Program(context, program_source);
		error_code = program.build(devices);
		CHECK_CL(error_code);
		
		// Create kernel.
		kernel = cl::Kernel(program, "row_band_select", &error_code);
		CHECK_CL(error_code);

		d_data = cl::Buffer(context, CL_MEM_READ_WRITE, m * n * sizeof(uint8_t));

	}


	//virtual void Process(Event<uint8_t>& event) = 0;
	void Process(Event<uint8_t>& event) {

		// Copy memory to device.
		error_code = queue.enqueueWriteBuffer(d_data, CL_TRUE, 0, event.m * event.n * sizeof(uint8_t), event.spectra.data());	
		CHECK_CL(error_code);

		// Set args.
		error_code = kernel.setArg(0, d_data);
		CHECK_CL(error_code);
		error_code = kernel.setArg(1, static_cast<unsigned int>(event.m));
		CHECK_CL(error_code);
		error_code = kernel.setArg(2, static_cast<unsigned int>(event.n));
		CHECK_CL(error_code);
		error_code = kernel.setArg(3, static_cast<float>(threshold));
		CHECK_CL(error_code);
		error_code = kernel.setArg(4, local_work_group_size * sizeof(float), NULL);
		CHECK_CL(error_code);
		error_code = kernel.setArg(5, local_work_group_size * sizeof(float), NULL);
		CHECK_CL(error_code);

		// Run kernel.
		error_code = queue.enqueueNDRangeKernel(kernel, cl::NullRange, event.m, local_work_group_size);
		CHECK_CL(error_code);

		// Copy device mem back to host.
		error_code = queue.enqueueReadBuffer(d_data, CL_TRUE, 0, event.m * event.n * sizeof(uint8_t), event.spectra.data());
		CHECK_CL(error_code);
	
	}

};


#endif
