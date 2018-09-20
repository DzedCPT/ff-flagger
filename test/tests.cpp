
#include <random>
#include <vector>
#include <climits>
#include <limits>
#include <iostream>
#include <functional>

#include "device.h"
//#include "smooth.h"
//#include "band_threshold.h"
//#include "sum_threshold.h"
//#include "band_sum_threshold.h"
//#include "edge_threshold.h"

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#define CHECK_VEC_EQUAL(x, y) \
	REQUIRE(x.size() == y.size()); \
	for (size_t i = 0; i < x.size(); ++i) { \
		if (x[i] != Approx(y[i]).margin(1e-6)) { \
			REQUIRE(x[i] == Approx(y[i]).margin(1e-6)); \
		} \
	}

using namespace std;
GPUEnviroment gpu;
std::vector<float> vec;
std::vector<float> results;
cl::Buffer d_in;
cl::Buffer d_out;
size_t m; 
size_t n;
size_t local_size;

std::random_device rd;     
//std::mt19937 rng(rd()); 
std::mt19937 rng(10); 
//std::uniform_int_distribution<T> uni;
std::uniform_real_distribution<> uni;

int RandInt(int min, int max) {
	return std::uniform_int_distribution<int>(min, max)(rng);
}

void InitExperiment(int max_m, int max_n = 1, float min_val = -1000, float max_val = 1000) {
	m = RandInt(1, max_m);
	n = RandInt(1, max_n);
	local_size = RandInt(0, 1000);
	uni = std::uniform_real_distribution<>(min_val, max_val);
	vec.resize(m * n);
	results.resize(m * n);
	
	for (auto& v: vec) { 
		v = round(uni(rng)); 
	}
		
	d_in = gpu.InitBuffer(CL_MEM_READ_WRITE , vec.size() *sizeof(float));
	d_out = gpu.InitBuffer(CL_MEM_READ_WRITE , vec.size() *sizeof(float));
	gpu.WriteToBuffer(vec.data(), d_in, vec.size() * sizeof(float));

}


TEST_CASE( "Test GPU Reduce.", "[reduce], [kernel]" ) {
	for (size_t test = 0; test < 10; test++) {
		InitExperiment(100000);

		// GPU.
		float result = gpu.Reduce(d_in, 103, m, local_size);

		// Sequential.
		float correct = std::accumulate(vec.begin(), vec.end(), 0.0);

		REQUIRE(correct == Approx(result));
	}
}


TEST_CASE( "Test GPU Transpose.", "[transpose], [kernel]" ) {
	for (size_t test = 0; test < 10; test++) {
		InitExperiment(1000, 1000, 0, 1);
		std::vector<float> correct(vec.size());

		// Sequential Implementation.
		for (size_t i = 0; i < m; i++) {
			for (size_t j = 0; j < n; j++) {
				correct[j * m + i]	= vec[i * n + j];
			
			}	
		}


		// GPU.
		gpu.Transpose(d_out, d_in, m, n, 25, 25);
		gpu.ReadFromBuffer(results.data(), d_out, m * n *sizeof(float));

		CHECK_VEC_EQUAL(correct, results);
	}
}


TEST_CASE( "Test Mask.", "[mask], [kernel]" ) {
	float mask_value = 10;
	for (size_t test = 0; test < 10; test++) {
		InitExperiment(1000, 1000, 0, 255);
		std::vector<float> mask(m * n);

		// Generate random mask.
		for (auto& v: mask) { v = (0.5 < std::uniform_real_distribution<>(0, 1)(rng)); } ;

		// Sequential Implementation.
		for (size_t i = 0; i < m; i++) {
			for (size_t j = 0; j < n; j++) {
				//vec[i * n + j] = vec[i * n + j]	* (1 - mask[i * n + j]);
				vec[i * n + j] = (mask[i * n + j] == 1) ? mask_value : vec[i * n + j];
			}
		}

		// GPU.	
		cl::Buffer d_mask = gpu.InitBuffer(CL_MEM_READ_WRITE, m * n * sizeof(float));
		gpu.WriteToBuffer(mask.data(), d_mask, m * n * sizeof(float));
		gpu.Mask(d_out, d_in, d_mask, mask_value,  m, n, 25, 25);
		gpu.ReadFromBuffer(results.data(), d_out, m * n *sizeof(float));

		CHECK_VEC_EQUAL(vec, results);
	}

}


TEST_CASE( "Test EdgeThreshold.", "[edge_threshold], [rfi]" ) {

	float threshold = 1;
	float median;
	float value;
	float window_stat;
	
	for (size_t test = 0; test < 10; test++) {
		InitExperiment(1000, 1000, 0, 256);
		
		// Compute MADs
		std::vector<float> mads(m);
		std::vector<float> mask(m * n, 0);
		std::vector<float> temp(vec);
		for (size_t i = 0; i < m; i++) {
			std::nth_element(temp.begin() + i * n, temp.begin() + i * n + n/2, temp.begin() + i * n + n);
			median = temp[i * n + n/2];
			std::transform(temp.begin() + i * n, temp.begin() + i * n + n, temp.begin() + i * n, [median](float x) -> float { return std::abs(x - median); });
			std::nth_element(temp.begin() + i * n, temp.begin() + i * n + n/2, temp.begin() + i * n + n);
			mads[i] = 1.4826 * temp[i * n + n/2];
		}
	
		for (size_t i = 0; i < m; i++) {
			for (size_t j = 1; j < n - 1; j++) {
				window_stat = vec[i * n + j];
				value = std::min(std::abs(window_stat - vec[i * n + j - 1]), std::abs(window_stat - vec[i * n + j + 1]));
				if (std::abs(value / mads[i]) > threshold) {
					mask[i * n + j]	= 1;
				}
	
			}
		}


		cl::Buffer gpu_mads = gpu.InitBuffer(CL_MEM_READ_WRITE , m * sizeof(float));
		cl::Buffer gpu_mask = gpu.InitBuffer(CL_MEM_READ_WRITE , m * n * sizeof(float));
		gpu.WriteToBuffer(mads.data(), gpu_mads, m * sizeof(float));
		gpu.EdgeThreshold(gpu_mask, gpu_mads, d_in, threshold, m, n, 12, 12);

		gpu.ReadFromBuffer(results.data(), gpu_mask, n * m * sizeof(float));
	
		CHECK_VEC_EQUAL(mask, results);

	}

}



TEST_CASE( "Test that MAD of each Row is correctly calculated..", "[mad], [rfi]" ) {

	float median;

	for (size_t test = 0; test < 10; test++) {
		InitExperiment(1000, 1000, 0, 255);

		// Sequential Implementation.
		std::vector<float> temp(vec);
		std::vector<float> cpu_mads(m);
		std::vector<float> cpu_medians(m);
		
		for (size_t i = 0; i < m; i++) {
			std::nth_element(temp.begin() + i * n, temp.begin() + i * n + n/2, temp.begin() + i * n + n);
			median = temp[i * n + n/2];
			cpu_medians[i] = median;
			std::transform(temp.begin() + i * n, temp.begin() + i * n + n, temp.begin() + i * n, [median](float x) -> float { return std::abs(x - median); });
			std::nth_element(temp.begin() + i * n, temp.begin() + i * n + n/2, temp.begin() + i * n + n);
			cpu_mads[i]  = 1.4826 * temp[i * n + n/2];
		}

		// Run on GPU.
		cl::Buffer b_gpu_medians = gpu.InitBuffer(CL_MEM_READ_WRITE, m * sizeof(float));
		gpu.MADRows(d_out, b_gpu_medians, d_in, m, n, local_size);
		std::vector<float> gpu_mads(m);
		std::vector<float> gpu_medians(m);
		gpu.ReadFromBuffer(gpu_mads.data(), d_out, m * sizeof(float));
		gpu.ReadFromBuffer(gpu_medians.data(), b_gpu_medians, m * sizeof(float));

	
		CHECK_VEC_EQUAL(cpu_mads, gpu_mads);
		CHECK_VEC_EQUAL(cpu_medians, gpu_medians);
	}
}


TEST_CASE( "Flag a row as RFI if it has above some threshold of RFI cells.", "[row_flagging], [rfi]" ) {

	float row_sum_threshold;
	int row_sum;

	for (size_t test = 0; test < 10; test++) {
		InitExperiment(1000, 1000, 0, 1);
		row_sum_threshold = n / 2;
		
		// Sequential Implementation.
		for (size_t i = 0; i < m; i++) {
			row_sum = std::accumulate(vec.begin() + i * n, vec.begin() + i * n + n, 0);
			if (row_sum > row_sum_threshold) { std::fill(vec.begin() + i * n, vec.begin() + i * n + n, 1); }
		}

		// Run on GPU.
		gpu.FlagRows(d_in, row_sum_threshold, m, n, local_size);
		gpu.ReadFromBuffer(results.data(), d_in, m * n * sizeof(float));


		CHECK_VEC_EQUAL(vec, results);

	}

}



