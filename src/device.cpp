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


// ********** Class setup functions  ********** // 


RFIPipeline::RFIPipeline (cl::Context& _context, 
		                  cl::CommandQueue& _queue, 
						  std::vector<cl::Device>& _devices, 
						  const Params& _params): params(_params)
{
	queue   = _queue;
	context = _context;
	devices = _devices;

	LoadKernels();
	InitMemBuffers(params.mode);

}


RFIPipeline::RFIPipeline (const Params& _params): params(_params)
{
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
	queue = cl::CommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE, &error_code); CHECK_CL(error_code);

	LoadKernels();
	InitMemBuffers(params.mode);
}


void RFIPipeline::LoadKernels (void) 
{
	// Read kernel file.
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
	reduce                 = cl::Kernel(program, "reduce", &error_code);                 CHECK_CL(error_code);
	mask_rows              = cl::Kernel(program, "mask_rows", &error_code);              CHECK_CL(error_code);
	transpose              = cl::Kernel(program, "transpose", &error_code);              CHECK_CL(error_code);
	compute_mads           = cl::Kernel(program, "compute_mads", &error_code);           CHECK_CL(error_code);
	sum_threshold          = cl::Kernel(program, "sum_threshold", &error_code);          CHECK_CL(error_code);
	detect_outliers        = cl::Kernel(program, "detect_outliers", &error_code);        CHECK_CL(error_code);
	scalar_division        = cl::Kernel(program, "scalar_division", &error_code);        CHECK_CL(error_code);
	compute_medians        = cl::Kernel(program, "compute_medians", &error_code);        CHECK_CL(error_code);
	compute_col_sums       = cl::Kernel(program, "compute_col_sums", &error_code);       CHECK_CL(error_code);
	flag_time_samples      = cl::Kernel(program, "flag_time_samples", &error_code);      CHECK_CL(error_code);
	compute_means_old      = cl::Kernel(program, "compute_means_old", &error_code);      CHECK_CL(error_code);
	compute_deviation      = cl::Kernel(program, "compute_deviation", &error_code);      CHECK_CL(error_code);
	replace_rfi_medians    = cl::Kernel(program, "replace_rfi_medians", &error_code);    CHECK_CL(error_code);
	replace_rfi_constant   = cl::Kernel(program, "replace_rfi_constant", &error_code);   CHECK_CL(error_code);
	point_edge_threshold   = cl::Kernel(program, "point_edge_threshold", &error_code);   CHECK_CL(error_code);
	mask_row_sum_threshold = cl::Kernel(program, "mask_row_sum_threshold", &error_code); CHECK_CL(error_code);


}


RFIPipeline::Params RFIPipeline::ReadConfigFile (const std::string config_file_name) {
	std::ifstream infile(config_file_name);
	std::string line;
	std::string token;
	std::string value;

	Params params;
	params.mode = 1;
	params.n_iter = 1;
	params.zero_dm_threshold = 2.5;
	params.edge_threshold = 3.5;

	while (std::getline(infile, line)) {
		try {
			if (line[0] == '#' || line.size() == 0) continue;

			value = line.substr(line.rfind(' '), line.size());
			token = line.substr(0, line.find(' '));
			if (token == "mode") params.mode = std::stoi(value);
			else if (token == "n_iter") params.n_iter = std::stoi(value);
			else if (token == "edge_threshold") params.edge_threshold = std::stof(value);
			else if (token == "zero_dm_threshold") params.zero_dm_threshold = std::stof(value);
			else {
				throw std::runtime_error(line);;
			}
		}
		catch (...) {
			std::cerr << "RFI config skipping line: " << line << std::endl;
		}
	}

	return params;
	

}


void RFIPipeline::InitMemBuffers (const int mode) 
{
	if (mode == 0) return;

	freq_medians = InitBuffer(CL_MEM_READ_WRITE, params.n_channels * sizeof(uint8_t));

	if (mode == 2 || mode == 3) {
		time_means = InitBuffer(CL_MEM_READ_WRITE, params.n_samples * sizeof(float));
		time_temp = InitBuffer(CL_MEM_READ_WRITE, params.n_samples * sizeof(float));
		flagged_samples = InitBuffer(CL_MEM_READ_WRITE, params.n_samples * sizeof(int));
		count = InitBuffer(CL_MEM_READ_WRITE, sizeof(int));
	}

	if (mode == 1 || mode == 3) {
		freq_mads = InitBuffer(CL_MEM_READ_WRITE, params.n_channels * sizeof(uint8_t));
	}
}


// ********** RFI mitigation pipelines ********** // 


void RFIPipeline::Flag (const cl::Buffer& data) 
{
	if (params.mode == 0) return;
    
    begin = std::chrono::high_resolution_clock::now(); 
	if (num_events % params.stat_freq == 0) {
		if (params.mode == 1 || params.mode == 3) {
			ComputeMads(freq_mads, freq_medians, data, params.n_channels, params.n_samples, params.n_padded_samples, 16, 16);
		}
		else {
			ComputeMedians(freq_medians, data, params.n_channels, params.n_samples, params.n_padded_samples, 16, 16);
		}
	}

	if (params.mode == 1 || params.mode == 3) {
		PointEdgeThreshold(data, freq_medians, freq_mads, params.edge_threshold, params.max_window_size, params.n_channels, params.n_samples, params.n_padded_samples, 1, 512);
	}

	if (params.mode == 2 || params.mode == 3) {
		ComputeMeans(time_means, data, params.n_channels, params.n_samples, params.n_padded_samples, 128, 1024, 4, 256);
		ClearBuffer(count, sizeof(int));
		ClearBuffer(flagged_samples, params.n_samples * sizeof(int));
		float mean, std;
		int num_flagged_samples = 0;
		for (int i = 0; i < params.n_iter; i++) {
			mean = FloatReduce(time_temp, time_means, params.n_samples) / (params.n_samples - num_flagged_samples);
			std  = ComputeStd(time_means, time_temp, num_flagged_samples, mean, params.n_samples, 1024);
			num_flagged_samples = DetectOutliers(flagged_samples, time_means, count, mean, std, params.zero_dm_threshold, params.n_samples, 128);
		}
		FlagTimeSamples(data, flagged_samples, freq_medians, num_flagged_samples, params.n_channels, params.n_samples, params.n_padded_samples, 128);

	}
	
	queue.finish();
    end = std::chrono::high_resolution_clock::now(); 
    time += std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count();
	num_events++;


}


// ********** GPU kernels  ********** // 


float RFIPipeline::FloatReduce (const cl::Buffer& d_out, 
		                        const cl::Buffer& d_in, 
								int n) 
{
	MARK_TIME(begin);
	const int n_threads = 128;
	int n_groups = (n + (n_threads * 2 - 1)) / (n_threads * 2);

	// First reduce step.
	CHECK_CL(reduce.setArg(0, d_out));
	CHECK_CL(reduce.setArg(1, d_in));
	CHECK_CL(reduce.setArg(2, n));
	CHECK_CL(reduce.setArg(3, n_threads * sizeof(float), NULL));
	CHECK_CL(queue.enqueueNDRangeKernel(reduce, cl::NullRange, n_groups * n_threads, n_threads));

	// Do rest of reduce on d_out so that d_in is left untouched.
	CHECK_CL(reduce.setArg(1, d_out));

	// Do remaining reduce steps.
	while (n_groups >= 2) {
		n = n_groups;	
		n_groups = (n + (n_threads * 2 - 1)) / (n_threads * 2);
	
		// Launch kernel for next reduce
	  	CHECK_CL(reduce.setArg(2, n));
	  	CHECK_CL(queue.enqueueNDRangeKernel(reduce, cl::NullRange, n_groups * n_threads, n_threads));
	
	}

	float to_return;
	ReadFromBuffer(&to_return, d_out, sizeof(float));

	ADD_TIME_SINCE(FloatReduce, begin);
	return to_return;


}


void RFIPipeline::Transpose (const cl::Buffer& d_out, 
		                     const cl::Buffer& d_in, 
						 	 int m, int n, 
							 int nx, int ny) 
{
	MARK_TIME(begin);
	CHECK_CL(transpose.setArg(0, d_out));
	CHECK_CL(transpose.setArg(1, d_in));
	CHECK_CL(transpose.setArg(2, m));
	CHECK_CL(transpose.setArg(3, n));
	CHECK_CL(transpose.setArg(4, 16 * 17 * sizeof(uint8_t), NULL));

	int n_threads_x = nx * ((m + nx - 1) / nx);
	int n_threads_y = ny * ((n + ny - 1) / ny);

	CHECK_CL(queue.enqueueNDRangeKernel(transpose, cl::NullRange, cl::NDRange(n_threads_x, n_threads_y), cl::NDRange(nx, ny)));
	ADD_TIME_SINCE(Transpose, begin);

}


void RFIPipeline::PointEdgeThreshold (const cl::Buffer& data, 
									  const cl::Buffer& medians, 
								      const cl::Buffer& mads, 
								      float threshold, 
								      int max_window_size, 
								      int m, int n, int N, 
								      int nx, int ny) 
{
	MARK_TIME(begin);
	
	CHECK_CL(point_edge_threshold.setArg(0, data));
	CHECK_CL(point_edge_threshold.setArg(1, medians));
	CHECK_CL(point_edge_threshold.setArg(2, mads));
	CHECK_CL(point_edge_threshold.setArg(3, threshold));
	CHECK_CL(point_edge_threshold.setArg(4, max_window_size));
	CHECK_CL(point_edge_threshold.setArg(5, m));
	CHECK_CL(point_edge_threshold.setArg(6, n));
	CHECK_CL(point_edge_threshold.setArg(7, N));
	CHECK_CL(point_edge_threshold.setArg(8, nx * (ny + max_window_size + 1) * sizeof(float), NULL));
	int n_threads_x = nx * ((m + nx - 1) / nx);
	int n_threads_y = ny;

	CHECK_CL(queue.enqueueNDRangeKernel(point_edge_threshold, cl::NullRange, cl::NDRange(n_threads_x, n_threads_y), cl::NDRange(nx, ny)));
	ADD_TIME_SINCE(PointEdgeThreshold, begin);

}


void RFIPipeline::SumThreshold (const cl::Buffer& m_out, 
		                        const cl::Buffer& d_in, 
							    const cl::Buffer& m_in, 
							    const cl::Buffer& medians, 
							    int num_iters, 
								float alpha,
							    int m, int n, int N,
							    int nx, int ny) 
{
		
	MARK_TIME(begin);
	// Compute thread layout.
	int n_threads_x = nx * ((m + nx - 1) / nx);
	int n_threads_y = ny * ((n + ny - 1) / ny);

	cl::Event evt;
	double time = 0.0;
	for (int window_size = 1, iter = 0; iter < num_iters; window_size*=2, iter++) {
		CHECK_CL(sum_threshold.setArg(0, m_out));
		CHECK_CL(sum_threshold.setArg(1, d_in));
		CHECK_CL(sum_threshold.setArg(2, m_in));
		CHECK_CL(sum_threshold.setArg(3, medians));
		CHECK_CL(sum_threshold.setArg(4, window_size));
		CHECK_CL(sum_threshold.setArg(5, iter));
		CHECK_CL(sum_threshold.setArg(6, alpha));
		CHECK_CL(sum_threshold.setArg(7, m));
		CHECK_CL(sum_threshold.setArg(8, n));
		CHECK_CL(sum_threshold.setArg(9, N));
		CHECK_CL(sum_threshold.setArg(10, nx * (ny + window_size) * sizeof(float), NULL));
		CHECK_CL(sum_threshold.setArg(11, nx * (ny + window_size) * sizeof(float), NULL));
		CHECK_CL(sum_threshold.setArg(12, nx * sizeof(float), NULL));
		
		CHECK_CL(queue.enqueueNDRangeKernel(sum_threshold, cl::NullRange, cl::NDRange(n_threads_x, n_threads_y), cl::NDRange(nx, ny), NULL, &evt));
		CopyBuffer (m_out, m_in, m * N * sizeof(uint8_t));
		evt.wait();
		time += evt.getProfilingInfo<CL_PROFILING_COMMAND_END>() -
		            evt.getProfilingInfo<CL_PROFILING_COMMAND_START>();
		
	}
	//std::cout << time / 1000 << std::endl;
	//std::swap(m_in, m_out);
	ADD_TIME_SINCE(SumThreshold, begin);


}


void RFIPipeline::SIRRankOperator (const cl::Buffer m_out,
						           const cl::Buffer m_in,
						           float density_ratio_threshold,
						           int m, int n, int N) 
{
	std::vector<float> psi(n);
	std::vector<float> psi_partial_sum(n);
	std::vector<int> psi_partial_min_index(n);
	std::vector<int> psi_partial_max_index(n);
	std::vector<uint8_t> host_mask(m * N);
	ReadFromBuffer(host_mask.data(), m_in, n * N * sizeof(uint8_t));

	for (int kk = 0; kk < m; kk++) {
		for (int j = 0; j < n; j++) {
			psi[j] = density_ratio_threshold - host_mask[kk * N + j];
		}
		psi_partial_sum[0] = 0;
		for (int j = 0; j < n; j++) {
			psi_partial_sum[j + 1] = psi_partial_sum[j] + psi[j];
		}


		psi_partial_min_index[0] = 0;
		for (int j = 1; j < n; j++) {
			psi_partial_min_index[j] = psi_partial_min_index[j - 1];

			if (psi_partial_sum[psi_partial_min_index[j]] > psi_partial_sum[j]) { 
				psi_partial_min_index[j] = j; 
			}

		}

		psi_partial_max_index[n - 1] = n;
		for (int j = n - 2; j >= 0 ; j--) {
			psi_partial_max_index[j] = psi_partial_max_index[j + 1];
			if (psi_partial_sum[psi_partial_max_index[j]] < psi_partial_sum[j + 1]) { 
				psi_partial_max_index[j] = j + 1; 
			}
		}

		for (int j = 0; j < n; j++) {
			if (psi_partial_sum[psi_partial_max_index[j]] - psi_partial_sum[psi_partial_min_index[j]] >= 0) {
				host_mask[j] = 0;    
			}    
			else {
				host_mask[j] = 1;    
			}

		}
	}
	WriteToBuffer(host_mask.data(), m_out, n * N * sizeof(uint8_t));


}


void RFIPipeline::ComputeMads (const cl::Buffer& mads, 
		                       const cl::Buffer& medians, 
							   const cl::Buffer& d_in, 
							   int m, int n, int N,
							   int nx, int ny) 
{
	MARK_TIME(begin);
	CHECK_CL(compute_mads.setArg(0, mads));
	CHECK_CL(compute_mads.setArg(1, medians));
	CHECK_CL(compute_mads.setArg(2, d_in));
	CHECK_CL(compute_mads.setArg(3, m));
	CHECK_CL(compute_mads.setArg(4, n));
	CHECK_CL(compute_mads.setArg(5, N));
	CHECK_CL(compute_mads.setArg(6, nx * 256 * sizeof(int), NULL));

	//CHECK_CL(queue.enqueueNDRangeKernel(compute_mads, cl::NullRange, nx * ((m + nx - 1) / nx), nx));
	int n_threads_x = nx * ((m + nx - 1) / nx);
	CHECK_CL(queue.enqueueNDRangeKernel(compute_mads, cl::NullRange, cl::NDRange(n_threads_x, ny), cl::NDRange(nx, ny)));
	ADD_TIME_SINCE(ComputeMads, begin);


}


void RFIPipeline::ComputeMedians (const cl::Buffer& medians, 
		                          const cl::Buffer& d_in, 
								  int m, int n, int N,
								  int nx, int ny) 
{

	MARK_TIME(begin);
	CHECK_CL(compute_medians.setArg(0, medians));
	CHECK_CL(compute_medians.setArg(1, d_in));
	CHECK_CL(compute_medians.setArg(2, m));
	CHECK_CL(compute_medians.setArg(3, n));
	CHECK_CL(compute_medians.setArg(4, N));
	CHECK_CL(compute_medians.setArg(5, nx * 256 * sizeof(int), NULL));

	//CHECK_CL(queue.enqueueNDRangeKernel(compute_medians, cl::NullRange, nx * ((m + nx - 1) / nx), nx));
	int n_threads_x = nx * ((m + nx - 1) / nx);
	CHECK_CL(queue.enqueueNDRangeKernel(compute_medians, cl::NullRange, cl::NDRange(n_threads_x, ny), cl::NDRange(nx, ny)));

	ADD_TIME_SINCE(ComputeMedians, begin);


}


void RFIPipeline::ComputeMeans (const cl::Buffer& d_out, 
		                        const cl::Buffer& d_in, 
								int m, int n, int N,
							    int lx, int ly,
								int nx, int ny) 
{

	// Set args.
	MARK_TIME(begin);
	CHECK_CL(compute_col_sums.setArg(0, d_out));
	CHECK_CL(compute_col_sums.setArg(1, d_in));
	CHECK_CL(compute_col_sums.setArg(2, lx));
	CHECK_CL(compute_col_sums.setArg(3, ly));
	CHECK_CL(compute_col_sums.setArg(4, m));
	CHECK_CL(compute_col_sums.setArg(5, n));
	CHECK_CL(compute_col_sums.setArg(6, N));
	CHECK_CL(compute_col_sums.setArg(7, ly * sizeof(int), NULL));
					
	// Run kernel.
	//CHECK_CL(queue.enqueueNDRangeKernel(compute_col_sums, cl::NullRange, global_range, local_range));
	int n_threads_x = nx * ((m + lx - 1) / lx);
	int n_threads_y = ny * ((n + ly - 1) / ly);
	CHECK_CL(queue.enqueueNDRangeKernel(compute_col_sums, cl::NullRange, cl::NDRange(n_threads_x, n_threads_y), cl::NDRange(nx, ny)));

	CHECK_CL(scalar_division.setArg(0, d_out));
	CHECK_CL(scalar_division.setArg(1, d_out));
	CHECK_CL(scalar_division.setArg(2, m));
	CHECK_CL(scalar_division.setArg(3, n));

	nx = 1024;
	n_threads_x = nx * ((n + lx - 1) / lx);
	CHECK_CL(queue.enqueueNDRangeKernel(scalar_division, cl::NullRange, n_threads_x, nx));
	
	ADD_TIME_SINCE(ComputeMeans, begin);

}


float RFIPipeline::ComputeStd (const cl::Buffer& d_in, 
		                       const cl::Buffer& temp, 
							   int n_flagged_samples,
							   float mean, 
							   int n, 
							   int nx) 
{
	MARK_TIME(begin);
	CHECK_CL(compute_deviation.setArg(0, temp));
	CHECK_CL(compute_deviation.setArg(1, d_in));
	CHECK_CL(compute_deviation.setArg(2, n_flagged_samples));
	CHECK_CL(compute_deviation.setArg(3, mean));
	CHECK_CL(compute_deviation.setArg(4, n));

	CHECK_CL(queue.enqueueNDRangeKernel(compute_deviation, cl::NullRange, nx * ((n + nx - 1) / nx), nx));

	float to_return = std::sqrt(FloatReduce(temp, temp, n));
	ADD_TIME_SINCE(ComputeStd, begin);
	return to_return;

}


int RFIPipeline::DetectOutliers (const cl::Buffer& d_out, 
		                         const cl::Buffer& d_in, 
							     const cl::Buffer& count,
							     float mean, 
								 float std, 
								 float threshold, 
								 int n, 
								 int nx) 
{
	MARK_TIME(begin);
	
	CHECK_CL(detect_outliers.setArg(0, d_out));
	CHECK_CL(detect_outliers.setArg(1, d_in));
	CHECK_CL(detect_outliers.setArg(2, count));
	CHECK_CL(detect_outliers.setArg(3, mean));
	CHECK_CL(detect_outliers.setArg(4, std));
	CHECK_CL(detect_outliers.setArg(5, threshold));
	CHECK_CL(detect_outliers.setArg(6, n));

	CHECK_CL(queue.enqueueNDRangeKernel(detect_outliers, cl::NullRange, nx * ((n + nx - 1) / nx), nx));
	ADD_TIME_SINCE(DetectOutliers, begin);
	int h_count;
	ReadFromBuffer(&h_count, count, sizeof(int));
	return h_count;

}

void RFIPipeline::MaskRowSumThreshold (const cl::Buffer& m_out, 
					  				   const cl::Buffer& m_in, 
									   int m, int n, int N)
{
	MARK_TIME(begin);

	CHECK_CL(mask_row_sum_threshold.setArg(0, m_out));
	CHECK_CL(mask_row_sum_threshold.setArg(1, m_in));
	CHECK_CL(mask_row_sum_threshold.setArg(2, m));
	CHECK_CL(mask_row_sum_threshold.setArg(3, n));
	CHECK_CL(mask_row_sum_threshold.setArg(4, N));
	CHECK_CL(mask_row_sum_threshold.setArg(5, 8 * 128 * sizeof(int), NULL));
					
	int n_threads_x = 8 * ((m + 8 - 1) / 8);
	CHECK_CL(queue.enqueueNDRangeKernel(mask_row_sum_threshold, cl::NullRange, cl::NDRange(n_threads_x, 128), cl::NDRange(8, 128)));
	ADD_TIME_SINCE(MaskSumThreshold, begin);

}


void RFIPipeline::MaskRows(const cl::Buffer& m_out, 
		                   const cl::Buffer& m_in, 
						   int m, int n, int N,
						   int nx, int ny) 
{
	MARK_TIME(begin);
	CHECK_CL(mask_rows.setArg(0, m_out));
	CHECK_CL(mask_rows.setArg(1, m_in));
	CHECK_CL(mask_rows.setArg(2, m));
	CHECK_CL(mask_rows.setArg(3, n));
	CHECK_CL(mask_rows.setArg(4, N));

	int n_threads_x = nx * ((m + nx - 1) / nx);
	CHECK_CL(queue.enqueueNDRangeKernel(mask_rows, cl::NullRange, cl::NDRange(n_threads_x, ny), cl::NDRange(nx, ny)));
	ADD_TIME_SINCE(MaskRows, begin);

}

void RFIPipeline::FlagTimeSamples (const cl::Buffer& d_out, 
			                       const cl::Buffer& m_in, 
								   const cl::Buffer& medians,
						           int num_flagged_samples,
				   		           int m, int n, int N,
								   int nx) 
{
	
	MARK_TIME(begin);
	CHECK_CL(flag_time_samples.setArg(0, d_out));
	CHECK_CL(flag_time_samples.setArg(1, m_in));
	CHECK_CL(flag_time_samples.setArg(2, medians));
	CHECK_CL(flag_time_samples.setArg(3, num_flagged_samples));
	CHECK_CL(flag_time_samples.setArg(4, m));
	CHECK_CL(flag_time_samples.setArg(5, n));
	CHECK_CL(flag_time_samples.setArg(6, N));

	int n_threads_x = nx * ((num_flagged_samples + nx - 1) / nx);
	CHECK_CL(queue.enqueueNDRangeKernel(flag_time_samples, cl::NullRange, n_threads_x, nx));


	ADD_TIME_SINCE(FlagTimeSamples, begin);


}



