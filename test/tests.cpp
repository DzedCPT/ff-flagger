
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
std::vector<uint8_t> vec;
std::vector<uint8_t> results;
cl::Buffer d_in;
cl::Buffer d_out;
size_t m; 
size_t n;
size_t local_size;

std::random_device rd;     
//std::mt19937 rng(rd()); 
std::mt19937 rng(5); 
//std::uniform_int_distribution<T> uni;
//std::uniform_real_distribution<> uni;
std::uniform_int_distribution<int> uni;

int RandInt(int min, int max) {
	return std::uniform_int_distribution<int>(min, max)(rng);
}

void InitExperiment(int max_m, int max_n = 1, float min_val = -1000, float max_val = 1000) {
	m = RandInt(1, max_m);
	n = RandInt(1, max_n);
	local_size = RandInt(0, 1000);
	uni = std::uniform_int_distribution<int>(min_val, max_val);
	vec.resize(m * n);
	results.resize(m * n);
	
	for (auto& v: vec) { 
		v = round(uni(rng)); 
	}
		
	d_in = gpu.InitBuffer(CL_MEM_READ_WRITE , vec.size() * sizeof(uint8_t));
	d_out = gpu.InitBuffer(CL_MEM_READ_WRITE , vec.size() * sizeof(uint8_t));
	gpu.WriteToBuffer(vec.data(), d_in, vec.size() * sizeof(uint8_t));

}

//float Mean(std::vector<float>& values, size_t len) {
template<typename Iterator>
float Mean(Iterator start, Iterator end) {
	return std::accumulate(start, end, 0.0) / std::distance(start, end);

}


//template<typename T>
template<typename Iterator>
float StandardDeviation(Iterator start, Iterator end, float mean) {
	float total_deviation = 0.0;
	size_t size = std::distance(start, end);
	for (auto iter = start; iter != end; iter++) {
		total_deviation += std::pow(mean - *iter, 2) / (size - 1);
	}

	return std::sqrt(total_deviation);
		
}
	

TEST_CASE( "Test GPU Reduce.", "[grubb], [kernel]" ) {
	size_t work_per_thread = 17;
	size_t step_size;
	for (size_t test = 0; test < 10; test++) {
		InitExperiment(100000, 1, 0, 255);
		work_per_thread = 17;
		local_size = 645;

		step_size = work_per_thread * local_size;
		step_size = std::min(step_size,  vec.size());

		for(auto iter = vec.begin(); iter != vec.end(); iter += step_size) {
			step_size = std::min(step_size, (size_t) std::distance(iter, vec.end()));
			float mean = Mean(iter, iter + step_size);
			float std = StandardDeviation(iter, iter + step_size, mean);
			std::transform(iter, iter + step_size, iter, [mean, std](uint8_t x) -> uint8_t { return std::abs(x - mean) > std; });
			
		}

		gpu.Grubb(d_in, m, work_per_thread, 1, local_size);

		gpu.ReadFromBuffer(results.data(), d_in, m * sizeof(uint8_t));
		
		CHECK_VEC_EQUAL(results, vec);

	}
}


TEST_CASE( "Test Flag rows.", "[row_flag], [rfi]" ) {
	for (size_t test = 0; test < 10; test++) {
		InitExperiment(1000, 1000, 0, 255);
		std::vector<uint8_t> mask(m);
		for (auto& v: mask) { v = (0.5 < std::uniform_real_distribution<>(0, 1)(rng)); } ;

		for (size_t i = 0; i < m; i++) {
			if (mask[i] == 1) {
				for (size_t j = 0; j < n; j++) {
					vec[i * n + j] = 0;
				}	
			}
		}

		// GPU.	
		cl::Buffer d_mask = gpu.InitBuffer(CL_MEM_READ_WRITE, m * sizeof(uint8_t));
		gpu.WriteToBuffer(mask.data(), d_mask, m * sizeof(uint8_t));
		gpu.MaskRows(d_in, d_mask, 0,  m, n, local_size);
		
		gpu.ReadFromBuffer(results.data(), d_in, m * n *sizeof(uint8_t));

		CHECK_VEC_EQUAL(vec, results);


	}

}



//TEST_CASE( "Test GPU Reduce.", "[reduce], [kernel]" ) {
	//for (size_t test = 0; test < 10; test++) {
		//InitExperiment(100000);

		//// GPU.
		//float result = gpu.Reduce(d_in, 103, m, local_size);

		//// Sequential.
		//float correct = std::accumulate(vec.begin(), vec.end(), 0.0);

		//REQUIRE(correct == Approx(result));
	//}
//}


TEST_CASE( "Test GPU Transpose.", "[transpose], [kernel]" ) {
	for (size_t test = 0; test < 10; test++) {
		InitExperiment(1000, 1000, 0, 256);
		std::vector<int> correct(vec.size());

		// Sequential Implementation.
		for (size_t i = 0; i < m; i++) {
			for (size_t j = 0; j < n; j++) {
				correct[j * m + i]	= vec[i * n + j];
			
			}	
		}

		// GPU.
		gpu.Transpose(d_out, d_in, m, n, 25, 25);
		gpu.ReadFromBuffer(results.data(), d_out, m * n * sizeof(uint8_t));

		CHECK_VEC_EQUAL(correct, results);
	}
}


TEST_CASE( "Test Mask.", "[mask], [kernel]" ) {
	uint8_t mask_value = 10;
	for (size_t test = 0; test < 10; test++) {
		InitExperiment(1000, 1000, 0, 255);
		std::vector<uint8_t> mask(m * n);

		// Generate random mask.
		for (auto& v: mask) { v = (0.5 < std::uniform_real_distribution<>(0, 1)(rng)); } ;

		// Sequential Implementation.
		for (size_t i = 0; i < m; i++) {
			for (size_t j = 0; j < n; j++) {
				vec[i * n + j] = (mask[i * n + j] == 1) ? mask_value : vec[i * n + j];
			}
		}

		// GPU.	
		cl::Buffer d_mask = gpu.InitBuffer(CL_MEM_READ_WRITE, m * n * sizeof(uint8_t));
		gpu.WriteToBuffer(mask.data(), d_mask, m * n * sizeof(uint8_t));
		gpu.Mask(d_out, d_in, d_mask, mask_value,  m, n, 25, 25);
		gpu.ReadFromBuffer(results.data(), d_out, m * n *sizeof(uint8_t));

		CHECK_VEC_EQUAL(vec, results);
	}

}


TEST_CASE( "Test EdgeThreshold.", "[edge_threshold], [rfi]" ) {

	float threshold = 1;
	uint8_t median;
	uint8_t value;
	uint8_t window_stat;
	
	for (size_t test = 0; test < 10; test++) {
		InitExperiment(1000, 1000, 0, 256);
		
		// Compute MADs
		std::vector<uint8_t> mads(m);
		std::vector<uint8_t> mask(m * n, 0);
		std::vector<uint8_t> temp(vec);
		for (size_t i = 0; i < m; i++) {
			std::nth_element(temp.begin() + i * n, temp.begin() + i * n + n/2, temp.begin() + i * n + n);
			median = temp[i * n + n/2];
			std::transform(temp.begin() + i * n, temp.begin() + i * n + n, temp.begin() + i * n, [median](uint8_t x) -> uint8_t { return std::abs(x - median); });
			std::nth_element(temp.begin() + i * n, temp.begin() + i * n + n/2, temp.begin() + i * n + n);
			mads[i] = temp[i * n + n/2];
		}
	
		for (size_t i = 0; i < m; i++) {
			for (size_t j = 1; j < n - 1; j++) {
				window_stat = vec[i * n + j];
				value = std::min(std::abs(window_stat - vec[i * n + j - 1]), std::abs(window_stat - vec[i * n + j + 1]));
				if (std::abs(value / (1.4826 * mads[i])) > threshold) {
					mask[i * n + j]	= 1;
				}
	
			}
		}

		cl::Buffer gpu_mads = gpu.InitBuffer(CL_MEM_READ_WRITE , m * sizeof(uint8_t));
		cl::Buffer gpu_mask = gpu.InitBuffer(CL_MEM_READ_WRITE , m * n * sizeof(uint8_t));
		gpu.WriteToBuffer(mads.data(), gpu_mads, m * sizeof(uint8_t));
		gpu.EdgeThreshold(gpu_mask, gpu_mads, d_in, threshold, m, n, 12, 12);

		gpu.ReadFromBuffer(results.data(), gpu_mask, n * m * sizeof(uint8_t));
	
		CHECK_VEC_EQUAL(mask, results);

	}

}



TEST_CASE( "Test that MAD of each Row is correctly calculated..", "[mad], [rfi]" ) {

	uint8_t median;

	for (size_t test = 0; test < 10; test++) {
		InitExperiment(1000, 1000, 0, 255);

		// Sequential Implementation.
		std::vector<uint8_t> temp(vec);
		std::vector<uint8_t> cpu_mads(m);
		std::vector<uint8_t> cpu_medians(m);
		
		for (size_t i = 0; i < m; i++) {
			std::nth_element(temp.begin() + i * n, temp.begin() + i * n + n/2, temp.begin() + i * n + n);
			median = temp[i * n + n/2];
			cpu_medians[i] = median;
			std::transform(temp.begin() + i * n, temp.begin() + i * n + n, temp.begin() + i * n, [median](float x) -> float { return std::abs(x - median); });
			std::nth_element(temp.begin() + i * n, temp.begin() + i * n + n/2, temp.begin() + i * n + n);
			cpu_mads[i]  = temp[i * n + n/2];
		}

		// Run on GPU.
		cl::Buffer b_gpu_medians = gpu.InitBuffer(CL_MEM_READ_WRITE, m * sizeof(uint8_t));
		gpu.MADRows(d_out, b_gpu_medians, d_in, m, n, local_size);
		std::vector<uint8_t> gpu_mads(m);
		std::vector<uint8_t> gpu_medians(m);
		gpu.ReadFromBuffer(gpu_mads.data(), d_out, m * sizeof(uint8_t));
		gpu.ReadFromBuffer(gpu_medians.data(), b_gpu_medians, m * sizeof(uint8_t));

	
		CHECK_VEC_EQUAL(cpu_mads, gpu_mads);
		CHECK_VEC_EQUAL(cpu_medians, gpu_medians);
	}
}


//TEST_CASE( "Flag a row as RFI if it has above some threshold of RFI cells.", "[row_flagging], [rfi]" ) {

	//float row_sum_threshold;
	//int row_sum;

	//for (size_t test = 0; test < 10; test++) {
		//InitExperiment(1000, 1000, 0, 1);
		//row_sum_threshold = n / 2;
		
		//// Sequential Implementation.
		//for (size_t i = 0; i < m; i++) {
			//row_sum = std::accumulate(vec.begin() + i * n, vec.begin() + i * n + n, 0);
			//if (row_sum > row_sum_threshold) { std::fill(vec.begin() + i * n, vec.begin() + i * n + n, 1); }
		//}

		//// Run on GPU.
		//gpu.FlagRows(d_in, row_sum_threshold, m, n, local_size);
		//gpu.ReadFromBuffer(results.data(), d_in, m * n * sizeof(float));


		//CHECK_VEC_EQUAL(vec, results);

	//}

//}



