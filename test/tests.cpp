
#include <random>
#include <vector>
#include <climits>
#include <limits>
#include <iostream>
#include <functional>

#include "device.h"
#include "timing.h"

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#define CHECK_VEC_EQUAL(x, y) \
	REQUIRE(x.size() == y.size()); \
	for (int i = 0; i < (int) x.size(); ++i) { \
		if (x[i] != Approx(y[i]).margin(1e-6)) { \
			REQUIRE(x[i] == Approx(y[i]).margin(1e-6)); \
		} \
	} 


using namespace std;

RFIPipeline::Params param;
RFIPipeline gpu(param);

// Host data.
std::vector<uint8_t> vec;
std::vector<uint8_t> results;
std::vector<float> float_vec;
std::vector<float> float_results;

// Device data.
cl::Buffer d_in;
cl::Buffer d_out;
cl::Buffer float_d_in;
cl::Buffer float_d_out;

int m; 
int n;
int N;
int local_size;

// Random number generation.
std::random_device rd;     
//std::mt19937 rng(rd()); 
std::mt19937 rng(5); 
std::uniform_int_distribution<int> uni;

int RandInt (int min, int max) {
	return std::uniform_int_distribution<int>(min, max)(rng);
}

void InitExperiment (int max_m, int max_n = 1, float min_val = -1000, float max_val = 1000) {
	m = RandInt(1, max_m);
	n = RandInt(1, max_n);
	N = n;
	if (N != 1) {
		N += RandInt(1, 200);
	}
	local_size = RandInt(50, 1000);
	uni = std::uniform_int_distribution<int>(min_val, max_val);
	vec.resize(m * N);
	results.resize(m * N);
	float_results.resize(m * N);
	
	for (auto& v: vec) { 
		v = round(uni(rng)); 
	}

	float_results.resize(m * N);
	float_vec.resize(m * N);
	std::copy(vec.begin(), vec.end(), float_vec.begin());
		
	d_in = gpu.InitBuffer(CL_MEM_READ_WRITE , vec.size() * sizeof(uint8_t));
	d_out = gpu.InitBuffer(CL_MEM_READ_WRITE , vec.size() * sizeof(uint8_t));
	float_d_in = gpu.InitBuffer(CL_MEM_READ_WRITE , vec.size() * sizeof(float));
	float_d_out = gpu.InitBuffer(CL_MEM_READ_WRITE , vec.size() * sizeof(float));
	gpu.WriteToBuffer(vec.data(), d_in, vec.size() * sizeof(uint8_t));
	gpu.WriteToBuffer(float_vec.data(), float_d_in, float_vec.size() * sizeof(float));

}

template<typename Iterator>
uint8_t Mad (Iterator start, Iterator end) 
{
	int n = std::distance(start, end);
	std::vector<uint8_t> temp(n);
	std::copy(start, end, temp.begin());
	std::nth_element(temp.begin(), temp.begin() + n/2, temp.end());
	uint8_t median = temp[n/2];
	std::transform(temp.begin(), temp.end(), temp.begin(), [median](uint8_t x) -> uint8_t { return std::abs(x - median); });
	std::nth_element(temp.begin(), temp.begin() + n/2, temp.end());
	return temp[n/2];

}

template<typename Iterator>
uint8_t Median (Iterator start, Iterator end) 
{
	int n = std::distance(start, end);
	std::vector<uint8_t> temp(n);
	std::copy(start, end, temp.begin());
	std::nth_element(temp.begin(), temp.begin() + n/2, temp.end());
	return temp[n/2];
}


template<typename Iterator>
float Mean (Iterator start, Iterator end) 
{
	return std::accumulate(start, end, 0.0) / std::distance(start, end);
}


template<typename Iterator>
float StandardDeviation (Iterator start, Iterator end, float mean) 
{
	float total_deviation = 0.0;
	int size = std::distance(start, end);
	for (auto iter = start; iter != end; iter++) {
		total_deviation += std::pow(mean - *iter, 2) / (size - 1);
	}
	return std::sqrt(total_deviation);
}


TEST_CASE( "Test: Reduce array of floats.", "[FloatReduce]" ) 
{
	for (int test = 0; test < 10; test++) {
		InitExperiment(10000);
			
		//std::vector<float> del(m);
		//for (auto& v: del) v = std::uniform_real_distribution<float>(0, 255)(rng);
	//uni = std::uniform_int_distribution<int>(min_val, max_val);
		//gpu.WriteToBuffer(del.data(), float_d_in, float_vec.size() * sizeof(float));
		float gpu_result = gpu.FloatReduce(float_d_out, float_d_in, m);
		float correct = std::accumulate(float_vec.begin(), float_vec.end(), 0.0);
		//float correct = std::accumulate(del.begin(), del.end(), 0.0);

		REQUIRE(correct == Approx(gpu_result));
	}
}


TEST_CASE( "Test: Transpose.", "[Transpose]" ) 
{

	for (int test = 0; test < 10; test++) {
		InitExperiment(1000, 1000, 0, 10);
		std::vector<int> correct(vec.size());

		// Sequential Implementation.
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < N; j++) {
				correct[j * m + i]	= vec[i * N + j];
			
			}	
		}

		// GPU.
		gpu.Transpose(d_out, d_in, m, N, 16, 16);
		gpu.ReadFromBuffer(results.data(), d_out, m * N * sizeof(uint8_t));

		CHECK_VEC_EQUAL(correct, results);
	}
}


TEST_CASE( "Test EdgeThreshold.", "[EdgeThreshold]" ) 
{

	int max_window_size = 5;
	float threshold = 1;
	float window_stat;
	float value;

	std::vector<uint8_t> mads;
	std::vector<uint8_t> mask;
	
	for (int test = 0; test < 10; test++) {
		InitExperiment(1000, 1000, 0, 256);
		
		// Resize vector for m and n.		
		mads.resize(m);
		mask.resize(m * N);
		std::fill(mask.begin(), mask.end(), 0);
		
		// Sequential EdgeThresholding.
		for (int window_size = 1; window_size <= max_window_size; window_size++) {
			for (int i = 0; i < m; i++) {
				mads[i] = Mad(vec.begin() + i * N, vec.begin() + (i * N) + n);
				for (int j = 1; j < n - window_size; j++) {
					window_stat = Mean(vec.begin() + i * N + j, vec.begin() + i * N + j + window_size);
					value = std::min(std::abs(window_stat - vec[i * N + j - 1]), std::abs(window_stat - vec[i * N + j + window_size]));
					if (std::abs(value / (1.4826 * mads[i])) > threshold) {
						std::fill(mask.begin() + i * N + j, mask.begin() + i * N + j + window_size, 1);
					}

				}
			}
		}

		// Copy memory to GPU.
		cl::Buffer gpu_mads = gpu.InitBuffer(CL_MEM_READ_WRITE , m * sizeof(uint8_t));
		cl::Buffer gpu_mask = gpu.InitBuffer(CL_MEM_READ_WRITE , m * N * sizeof(uint8_t));
		cl::Buffer gpu_mask_out = gpu.InitBuffer(CL_MEM_READ_WRITE , m * N * sizeof(uint8_t));
		gpu.WriteToBuffer(mads.data(), gpu_mads, m * sizeof(uint8_t));

		// Parallel EdgeThresholding.
		gpu.EdgeThreshold(gpu_mask, d_in, gpu_mads, threshold, max_window_size, m, n, N, 32, 32);

		gpu.ReadFromBuffer(results.data(), gpu_mask, m * N * sizeof(uint8_t));
	
		CHECK_VEC_EQUAL(mask, results);

	}

}


TEST_CASE( "Test SumThreshold.", "[SumThreshold]" ) 
{

	int max_window_size = 5;
	std::vector<uint8_t> mask;
	std::vector<uint8_t> mask_out;
	std::vector<uint8_t> thresholds;
	float alpha = 6;
	float threshold;

	
	for (int test = 0; test < 10; test++) {
		InitExperiment(1000, 1000, 0, 256);
		
		// Compute MADs
		mask.resize(m * N);
		mask_out.resize(m * N);
		thresholds.resize(m);
		std::fill(mask.begin(), mask.end(), 0);
		std::fill(mask_out.begin(), mask_out.end(), 0);

		for (auto& v : thresholds) v = RandInt(0, 255);
	
		for (int window_size = 1, iter = 0; window_size <= max_window_size; window_size *= 2, iter++) { 
			for (int i = 0; i < m; i++) {

				float window_sum   = 0;
				int window_count = 0;
				int j;
				for (j = 0; j < window_size; j++) {
					if (mask[i * N + j] != 1) {
						window_count += 1;
						window_sum   += vec[i * N + j];
					}
				}
				threshold =  alpha * ((float) std::pow(1.5, iter) / window_size) + thresholds[i];
				
				if (window_sum > threshold * window_count) {
						std::fill(mask_out.begin() + i * N, mask_out.begin() + i * N + window_size, 1);
				}
				for ( ; j < n; j++) {
					if (mask[i * N + j] != 1) {
						window_count += 1;
						window_sum   += vec[i * N + j];
					}
					if (mask[i * N + j - window_size] != 1) {
						window_sum   -= vec[i * N + j - window_size];
						window_count -= 1;
					}

					 //Check if current window should be masked.
					 if (window_sum > threshold * window_count) {
						 std::fill(mask_out.begin() + i * N + j - window_size + 1, mask_out.begin() + i * N + j + 1, 1);
					 }
				
				}
				std::copy(mask_out.begin(), mask_out.end(), mask.begin());

			}
		}
		
		
		cl::Buffer gpu_mask = gpu.InitBuffer(CL_MEM_READ_WRITE , m * N * sizeof(uint8_t));
		cl::Buffer gpu_mask_out = gpu.InitBuffer(CL_MEM_READ_WRITE , m * N * sizeof(uint8_t));
		cl::Buffer gpu_thresholds = gpu.InitBuffer(CL_MEM_READ_WRITE , m * sizeof(uint8_t));
		gpu.ClearBuffer (gpu_mask, m * N * sizeof(uint8_t));
		gpu.ClearBuffer (gpu_mask_out, m * N * sizeof(uint8_t));
		gpu.WriteToBuffer(thresholds.data(), gpu_thresholds, m * sizeof(uint8_t));
		gpu.SumThreshold(gpu_mask_out, d_in, gpu_mask, gpu_thresholds, max_window_size, alpha,  m, n, N, 16, 16);

		gpu.ReadFromBuffer(results.data(), gpu_mask_out, m * N * sizeof(uint8_t));

		//for (int i = 0; i < m; i++) {
			//for (int j = 0; j < n ; j++) {
				//if (mask_out[i * N + j] != results[i * N + j]) cout << i << "problem" <<j << (int) mask_out[i * N + j]<< endl;
				////cout << (int) vec[i * N + j] << " ";
			//}
			////cout << endl;
		//}
		//for (int i = 0; i < m; i++) {
			//for (int j = 0; j < n ; j++) {
				//cout << (int) mask_out[i * N + j] << " ";
			//}
			//cout << endl;
		//}
		//cout << endl;
		//for (int i = 0; i < m; i++) {
			//for (int j = 0; j < n ; j++) {
				//cout << (int) results[i * N + j] << " ";
			//}
			//cout << endl;
		//}
		CHECK_VEC_EQUAL(mask_out, results);

	}

}


TEST_CASE( "Test: Compute  MAD of each row.", "[ComputeMads]" ) 
{

	for (int test = 0; test < 10; test++) {
		InitExperiment(1000, 1000, 0, 255);

		// Sequential Implementation.
		std::vector<uint8_t> cpu_mads(m);
		std::vector<uint8_t> cpu_medians(m);
		std::vector<uint8_t> gpu_mads(m);
		std::vector<uint8_t> gpu_medians(m);
		
		for (int i = 0; i < m; i++) {
			cpu_mads[i] = Mad(vec.begin() + i * N, vec.begin() + (i * N) + n);
			cpu_medians[i] = Median(vec.begin() + i * N, vec.begin() + (i * N) + n);
		}

		// Run on GPU.
		cl::Buffer b_gpu_medians = gpu.InitBuffer(CL_MEM_READ_WRITE, m * sizeof(uint8_t));
		cl::Buffer b_gpu_mads = gpu.InitBuffer(CL_MEM_READ_WRITE, m * sizeof(uint8_t));
		gpu.ComputeMads(b_gpu_mads, b_gpu_medians, d_in, m, n, N, 16, 2);
		gpu.ReadFromBuffer(gpu_mads.data(), b_gpu_mads, m * sizeof(uint8_t));
		gpu.ReadFromBuffer(gpu_medians.data(), b_gpu_medians, m * sizeof(uint8_t));

	
		CHECK_VEC_EQUAL(cpu_mads, gpu_mads);
		CHECK_VEC_EQUAL(cpu_medians, gpu_medians);
	}
}


TEST_CASE( "Test: Compute median of each row.", "[ComputeMedians]" ) 
{

	for (int test = 0; test < 10; test++) {
		InitExperiment(1000, 1000, 0, 255);

		// Sequential Implementation.
		std::vector<uint8_t> cpu_medians(m);
		results.resize(m);
		
		for (int i = 0; i < m; i++) {
			cpu_medians[i] = Median(vec.begin() + i * N, vec.begin() + (i * N) + n);
		}

		// Run on GPU.
		cl::Buffer gpu_medians = gpu.InitBuffer(CL_MEM_READ_WRITE, m * sizeof(uint8_t));
		gpu.ComputeMedians(gpu_medians, d_in, m, n, N, 16, 2);
		gpu.ReadFromBuffer(results.data(), gpu_medians, m * sizeof(uint8_t));

		CHECK_VEC_EQUAL(cpu_medians, results);
	}

}


TEST_CASE( "Test: Compute mean of each row.", "[ComputeMeans]" ) 
{
	cl::Buffer d_out_float;
	std::vector<float> gpu_results; 
	for (int test = 0; test < 10; test++) {
		InitExperiment(10000, 10000, 0, 255);
			
		gpu_results.resize(n);
		std::vector<float> col_sums(n, 0);
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				col_sums[j] += vec[i * N + j];
			}	
		}
		for (int j = 0; j < n; j++) {
			col_sums[j] /= m;
		}	

	
		d_out_float = gpu.InitBuffer(CL_MEM_READ_WRITE, n * sizeof(float));
		gpu.ClearBuffer(d_out_float, n * sizeof(float));
		gpu.ComputeMeans(d_out_float, d_in, m, n, N, 256, 256, 16, 16);
		gpu.ReadFromBuffer(gpu_results.data(), d_out_float, n * sizeof(float));

		CHECK_VEC_EQUAL(col_sums, gpu_results);
	}

}


TEST_CASE( "Test: Compute Standard Deviation.", "[ComputeStd]" ) 
{
	cl::Buffer temp;
	for (int test = 0; test < 10; test++) {
		if (test % 2 == 0) InitExperiment(10000, 1, 0, 255);
		
		float mean = Mean(float_vec.begin(), float_vec.end());
		float std = StandardDeviation(float_vec.begin(), float_vec.end(), mean);

		temp = gpu.InitBuffer(CL_MEM_READ_WRITE, m * sizeof(float));
		
		float gpu_std = gpu.ComputeStd(float_d_in, temp, mean,  m, 128);

		REQUIRE(gpu_std == Approx(std).margin(1e-6));

	}

}


TEST_CASE( "Test: Create outlier mask using mean and std.", "[DetectOutliers]" ) 
{
	cl::Buffer temp;
	for (int test = 0; test < 10; test++) {
		InitExperiment(10000, 1, 0, 255);
		
		// Sequential.
		float mean = Mean(float_vec.begin(), float_vec.end());
		float std = StandardDeviation(float_vec.begin(), float_vec.end(), mean);
		std::transform(float_vec.begin(), float_vec.end(), float_vec.begin(), [mean, std](float x) -> float { return std::abs(x - mean) > std; });

		// Parallel.
		gpu.DetectOutliers(float_d_out, float_d_in, mean, std, 1, m, local_size);
		gpu.ReadFromBuffer(float_results.data(), float_d_out, m * sizeof(float));
		CHECK_VEC_EQUAL(float_vec, float_results);

	}
}


TEST_CASE( "Test: ", "[MaskRowThreshold]" ) 
{
	for (int test = 0; test < 10; test++) {
		InitExperiment(1000, 1000, 0, 255);

		std::vector<uint8_t> mask(m * N);

		// Generate random mask.
		for (auto& v: mask) { v = (0.5 < std::uniform_real_distribution<>(0, 1)(rng)); } ;

		cl::Buffer d_mask = gpu.InitBuffer(CL_MEM_READ_WRITE, m * N * sizeof(uint8_t));
		gpu.WriteToBuffer(mask.data(), d_mask, m * N * sizeof(uint8_t));

		// Sequential Implementation.
		for (int i = 0; i < m; i++) {
			if (std::accumulate(mask.begin() + i * N, mask.begin() + (i * N) + n, 0) > 500) {
				std::fill(mask.begin() + i * N, mask.begin() + (i * N) + n, 1);
			}	
			
		}

		// GPU.	
		gpu.MaskRowSumThreshold(d_mask, d_mask, m, n, N);
		gpu.ReadFromBuffer(results.data(), d_mask, m * N * sizeof(uint8_t));
		CHECK_VEC_EQUAL(results, mask);

	}

}



TEST_CASE( "Test: Set masked rows to 1s.", "[MaskRows]" ) 
{
	for (int test = 0; test < 10; test++) {
		InitExperiment(1000, 1000, 0, 255);
		std::vector<float> mask(m);

		for (int i = 0; i < m; i++) {
			mask[i] =  (0.5 < std::uniform_real_distribution<>(0, 1)(rng));
			if (mask[i] == 1) {
				std::fill(vec.begin() + i * N, vec.begin() + (i * N) + n, 1);
			}
		}

		// GPU.	
		cl::Buffer d_mask = gpu.InitBuffer(CL_MEM_READ_WRITE, m * sizeof(float));
		gpu.WriteToBuffer(mask.data(), d_mask, m * sizeof(float));
		gpu.MaskRows(d_in, d_mask, m, n, N, 32, 32);
		
		gpu.ReadFromBuffer(results.data(), d_in, m * N *sizeof(uint8_t));

		CHECK_VEC_EQUAL(vec, results);


	}

}


TEST_CASE( "Test: Replace masked values with row median.", "[ReplaceRFIMedians]" ) 
{
	for (int test = 0; test < 10; test++) {
		InitExperiment(1000, 1000, 0, 255);

		std::vector<uint8_t> mask(m * N);
		std::vector<uint8_t> medians(m);

		// Generate random mask.
		for (auto& v: mask) { v = (0.5 < std::uniform_real_distribution<>(0, 1)(rng)); } ;
		for (auto& v: medians) { v = RandInt(0, 10); } ;

		// Sequential Implementation.
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				vec[i * N + j] = (mask[i * N + j] == 1) ? medians[i] : vec[i * N + j];
			}
		}

		// GPU.	
		cl::Buffer d_mask = gpu.InitBuffer(CL_MEM_READ_WRITE, m * N * sizeof(uint8_t));
		gpu.WriteToBuffer(mask.data(), d_mask, m * N * sizeof(uint8_t));
		cl::Buffer d_medians = gpu.InitBuffer(CL_MEM_READ_WRITE, m * sizeof(uint8_t));
		gpu.WriteToBuffer(medians.data(), d_medians, m * sizeof(uint8_t));
		gpu.ReplaceRFI(d_out, d_in, d_mask, d_medians, RFIPipeline::RFIReplaceMode::MEDIANS, m, n, N, 25, 25);
		gpu.ReadFromBuffer(results.data(), d_out, m * N *sizeof(uint8_t));
		bool passed = true;
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (vec[i * N + j] != results[i * N + j]) {
					passed = false;
				}
			}
		}

		REQUIRE(passed);

	}

}

TEST_CASE( "Test: Replace masked values with 0", "[ReplaceRFIConstant]" ) 
{
	for (int test = 0; test < 10; test++) {
		InitExperiment(1000, 1000, 0, 255);

		std::vector<uint8_t> mask(m * N);

		// Generate random mask.
		for (auto& v: mask) { v = (0.5 < std::uniform_real_distribution<>(0, 1)(rng)); } ;

		// Sequential Implementation.
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				vec[i * N + j] = (mask[i * N + j] == 1) ? 0 : vec[i * N + j];
			}
		}

		// GPU.	
		cl::Buffer d_mask = gpu.InitBuffer(CL_MEM_READ_WRITE, m * N * sizeof(uint8_t));
		cl::Buffer d_medians = gpu.InitBuffer(CL_MEM_READ_WRITE, m * sizeof(uint8_t));
		gpu.WriteToBuffer(mask.data(), d_mask, m * N * sizeof(uint8_t));
		gpu.ReplaceRFI(d_out, d_in, d_mask, d_medians, RFIPipeline::RFIReplaceMode::ZEROS, m, n, N, 25, 25);
		gpu.ReadFromBuffer(results.data(), d_out, m * N *sizeof(uint8_t));
		bool passed = true;
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++) {
				if (vec[i * N + j] != results[i * N + j]) {
					passed = false;
				}
			}
		}

		REQUIRE(passed);

	}

}


//TEST_CASE( "Test: Replace masked values with 0", "[ReplaceRFIConstant]" ) 
//{
	//for (int test = 0; test < 10; test++) {

		//std::vector<uint8_t> mask(m * N);

		//// Generate random mask.
		//for (auto& v: mask) { v = 0;} 

		//// Sequential Implementation.
		//for (int i = 0; i < m; i++) {
			//for (int j = 0; j < n; j++) {
				//vec[i * N + j] = (mask[i * N + j] == 1) ? 0 : vec[i * N + j];
			//}
		//}

		//// GPU.	
		//cl::Buffer d_mask = gpu.InitBuffer(CL_MEM_READ_WRITE, m * N * sizeof(uint8_t));
		//gpu.WriteToBuffer(mask.data(), d_mask, m * N * sizeof(uint8_t));
		////gpu.WriteToBuffer(mask.data(), d_mask, m * N * sizeof(uint8_t));
		////gpu.ReplaceRFI(d_out, d_in, d_mask, d_mask, RFIPipeline::RFIReplaceMode::ZEROS, m, n, N, 32, 32);
		////gpu.ReadFromBuffer(results.data(), d_out, m * N * sizeof(uint8_t));
		////bool passed = true;
		////for (int i = 0; i < m; i++) {
			////for (int j = 0; j < n; j++) {
				////if (vec[i * N + j] != results[i * N + j]) {
					////passed = false;
				////}
			////}
		////}

		////REQUIRE(passed);
	//}

//}



//TEST_CASE( "Test: Replace masked values with 0", "[ReplaceRFIConstant]" ) 
//{
	//for (int test = 0; test < 10; test++) {
		//InitExperiment(1000, 1000, 0, 255);

		//std::vector<uint8_t> mask(m * N);

		//// Generate random mask.
		//for (auto& v: mask) { v = 0; } ;

		//// Sequential Implementation.
		//for (int i = 0; i < m; i++) {
			//for (int j = 0; j < n; j++) {
				//vec[i * N + j] = (mask[i * N + j] == 1) ? 0 : vec[i * N + j];
			//}
		//}

		//// GPU.	
		//cl::Buffer d_mask = gpu.InitBuffer(CL_MEM_READ_WRITE, m * N * sizeof(uint8_t));
		//gpu.WriteToBuffer(mask.data(), d_mask, m * N * sizeof(uint8_t));
		//gpu.ReplaceRFI(d_out, d_in, d_mask, d_mask, RFIPipeline::RFIReplaceMode::ZEROS, m, n, N, 32, 32);
		//gpu.ReadFromBuffer(results.data(), d_out, m * N * sizeof(uint8_t));
		//bool passed = true;
		//for (int i = 0; i < m; i++) {
			//for (int j = 0; j < n; j++) {
				//if (vec[i * N + j] != results[i * N + j]) {
					//passed = false;
				//}
			//}
		//}

		//REQUIRE(passed);
	//}

//}


	//TEST_CASE( "Test Remove Means.", "[means], [kernel]" ) {
	////int work_per_thread = 17;
	////int step_size;
	//int block_size = 1000;
	//std::vector<float> means(block_size);
	//for (int test = 0; test < 10; test++) {
		//InitExperiment(10000, 10000, 0, 255);
		//std::vector<uint8_t> mask(n * m, 0);

		////work_per_thread = 17;
		////local_size = 645;
		//for (int block_start = 0; block_start != m; block_start += block_size) {
			//// Compute the means
			//block_size = std::min(block_size, m  - block_start);
			//means.resize(block_size);
			//for (int i = block_start, mean_index = 0; i < block_start + block_size; i++, mean_index++) {
				//means[mean_index] = std::accumulate(vec.begin() + i * n, vec.begin() + (i + 1) * n, 0.0) / n;
			//}
			//float mean = Mean(means.begin(), means.end());
			//float std = StandardDeviation(means.begin(), means.end(), mean);
			//std::transform(means.begin(), means.end(), means.begin(), [mean, std](uint8_t x) -> uint8_t { return std::abs(x - mean) > std; });
			//for (int i = block_start, mean_index = 0; i < block_start + block_size; i++, mean_index++) {
				//if (means[mean_index] == 0) {
					//std::fill(mask.begin() + i * n, mask.begin() + (i + 1) * n, 1) ;
				
				//}
			//}
			
			
		//}

		////step_size = work_per_thread * local_size;
		////step_size = std::min(step_size,  vec.size());

		////for(auto iter = vec.begin(); iter != vec.end(); iter += step_size) {
			////step_size = std::min(step_size, (int) std::distance(iter, vec.end()));
			////float mean = Mean(iter, iter + step_size);
			////float std = StandardDeviation(iter, iter + step_size, mean);
			////std::transform(iter, iter + step_size, iter, [mean, std](uint8_t x) -> uint8_t { return std::abs(x - mean) > std; });
			
		////}

		////gpu.OutlierDetection(d_in, m, work_per_thread, 1, local_size);

		////gpu.ReadFromBuffer(results.data(), d_in, m * sizeof(uint8_t));
		
		////CHECK_VEC_EQUAL(results, vec);

	//}
//}



//TEST_CASE( "Test Remove Means.", "[means], [kernel]" ) {
	//for (int test = 0; test < 1; test++) {
		//InitExperiment(10000, 10000, 0, 255);
		//std::vector<float> means(m);

		//for (int i = 0; i < m; i++) {
			//means[i] = std::accumulate(vec.begin() + i * n, vec.begin() + (i + 1) * n, 0.0) / n;	
		//}
				////gpu.OutlierDetection(d_in, m, work_per_thread, 1, local_size);

		////gpu.ReadFromBuffer(results.data(), d_in, m * sizeof(uint8_t));
		
		////CHECK_VEC_EQUAL(results, vec);

	//}
//}




////TEST_CASE( "Test GPU Reduce.", "[grubb], [kernel]" ) {
	////int work_per_thread = 17;
	////int step_size;
	////for (int test = 0; test < 10; test++) {
		////InitExperiment(100000, 1, 0, 255);
		////work_per_thread = 17;
		////local_size = 645;

		////step_size = work_per_thread * local_size;
		////step_size = std::min(step_size,  vec.size());

		////for(auto iter = vec.begin(); iter != vec.end(); iter += step_size) {
			////step_size = std::min(step_size, (int) std::distance(iter, vec.end()));
			////float mean = Mean(iter, iter + step_size);
			////float std = StandardDeviation(iter, iter + step_size, mean);
			////std::transform(iter, iter + step_size, iter, [mean, std](uint8_t x) -> uint8_t { return std::abs(x - mean) > std; });
			
		////}

		////gpu.OutlierDetection(d_in, m, work_per_thread, 1, local_size);

		////gpu.ReadFromBuffer(results.data(), d_in, m * sizeof(uint8_t));
		
		////CHECK_VEC_EQUAL(results, vec);

	////}
////}


//TEST_CASE( "Test Flag rows.", "[row_flag], [rfi]" ) {
	//for (int test = 0; test < 10; test++) {
		//InitExperiment(1000, 1000, 0, 255);
		//std::vector<uint8_t> mask(m);
		//for (auto& v: mask) { v = (0.5 < std::uniform_real_distribution<>(0, 1)(rng)); } ;
		//std::vector<uint8_t> medians(n);
		//for (auto& v: medians) { v = RandInt(0, 255); } ;

		//for (int i = 0; i < m; i++) {
			//if (mask[i] == 1) {
				//for (int j = 0; j < n; j++) {
					//vec[i * n + j] = medians[j];
				//}	
			//}
		//}

		//// GPU.	
		//cl::Buffer d_mask = gpu.InitBuffer(CL_MEM_READ_WRITE, m * sizeof(uint8_t));
		//gpu.WriteToBuffer(mask.data(), d_mask, m * sizeof(uint8_t));
		//cl::Buffer d_medians = gpu.InitBuffer(CL_MEM_READ_WRITE, n * sizeof(uint8_t));
		//gpu.WriteToBuffer(medians.data(), d_medians, n * sizeof(uint8_t));
		//gpu.MaskRows(d_in, d_mask, d_medians,  m, n, local_size);
		
		//gpu.ReadFromBuffer(results.data(), d_in, m * n *sizeof(uint8_t));

		//CHECK_VEC_EQUAL(vec, results);


	//}

//}


//TEST_CASE( "Test Mask.", "[mask], [kernel]" ) {
	//for (int test = 0; test < 10; test++) {
		//InitExperiment(1000, 1000, 0, 255);

		//std::vector<uint8_t> mask(m * n);
		//std::vector<uint8_t> medians(m);

		//// Generate random mask.
		//for (auto& v: mask) { v = (0.5 < std::uniform_real_distribution<>(0, 1)(rng)); } ;
		//for (auto& v: medians) { v = RandInt(0, 10); } ;

		//// Sequential Implementation.
		//for (int i = 0; i < m; i++) {
			//for (int j = 0; j < n; j++) {
				//vec[i * n + j] = (mask[i * n + j] == 1) ? medians[i] : vec[i * n + j];
			//}
		//}

		//// GPU.	
		//cl::Buffer d_mask = gpu.InitBuffer(CL_MEM_READ_WRITE, m * n * sizeof(uint8_t));
		//gpu.WriteToBuffer(mask.data(), d_mask, m * n * sizeof(uint8_t));
		//cl::Buffer d_medians = gpu.InitBuffer(CL_MEM_READ_WRITE, m * sizeof(uint8_t));
		//gpu.WriteToBuffer(medians.data(), d_medians, m * sizeof(uint8_t));
		//gpu.ReplaceRFI(d_out, d_in, d_mask, d_medians,  m, n, 25, 25);
		//gpu.ReadFromBuffer(results.data(), d_out, m * n *sizeof(uint8_t));

		//CHECK_VEC_EQUAL(vec, results);
	//}

//}



TEST_CASE( "Tsfsfdedfst EdgeThreshold.", "[del2]" ) {

	float threshold = 1;
	int window_size = 3;
	
	m = 1536;
	n = 43657;
	N = n;
	
	cl::Buffer d_data = gpu.InitBuffer(CL_MEM_READ_WRITE, m * n);
	cl::Buffer d_in = gpu.InitBuffer(CL_MEM_READ_WRITE, n  * sizeof(float));
	cl::Buffer d_medians = gpu.InitBuffer(CL_MEM_READ_WRITE, m * sizeof(float));
	std::vector<float> data(n);
	for (auto& v: data) {v = RandInt(0,255);}
	float mean = Mean(data.begin(), data.end());
	float std = StandardDeviation(data.begin(), data.end(), mean);
	gpu.WriteToBuffer(data.data(), d_in, n * sizeof(float));
	
		
	for (int test = 0; test < 5; test++) {
		//gpu.DetectOutliers (d_in, d_in, mean, std, threshold, n, 512);

	}
	
	gpu.queue.finish();
	gpu.queue.flush();

	auto begin = std::chrono::high_resolution_clock::now();
	for (int test = 0; test < 1; test++) {
		gpu.DetectOutliers (d_in, d_in, mean, std, threshold, n, 512);
		gpu.ReplaceRFI (d_data, d_data, d_in, d_medians, RFIPipeline::RFIReplaceMode::MEDIANS,m, n, N,16, 16);
	
	}
	gpu.queue.finish();
	auto end = std::chrono::high_resolution_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1  << std::endl;

	//m = 43657;
	//n = 1536;
	//N = n;
	

	//begin = std::chrono::high_resolution_clock::now();
	//for (int test = 0; test < 1000; test++) {
		//gpu.ComputeMeansOld(d_out, d_in, m, n, N);
	//}
	//gpu.queue.finish();
	//end = std::chrono::high_resolution_clock::now();
	//std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000  << std::endl;


}



TEST_CASE( "Tedfst EdgeThreshold.", "[del]" ) {

	float threshold = 1;
	int window_size = 3;
	
	n = 43657;
	m = 1536;
	N = n;
	
	cl::Buffer d_in = gpu.InitBuffer(CL_MEM_READ_WRITE , m * n  * sizeof(uint8_t));
	cl::Buffer d_out = gpu.InitBuffer(CL_MEM_READ_WRITE , n * sizeof(float));
		
	for (int test = 0; test < 5; test++) {
		gpu.ComputeMeans(d_out, d_in, m, n, N, 256, 256, 16, 16);
	}
	
	gpu.queue.finish();
	gpu.queue.flush();

	auto begin = std::chrono::high_resolution_clock::now();
	for (int test = 0; test < 1000; test++) {
		//gpu.ComputeMeans(d_out, d_in, m, n, N, 128, 1024, 4, 256);
		gpu.FloatReduce(d_in, d_in, n);
	}
	gpu.queue.finish();
	auto end = std::chrono::high_resolution_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000  << std::endl;

	m = 43657;
	n = 1536;
	N = n;
	

	begin = std::chrono::high_resolution_clock::now();
	for (int test = 0; test < 1000; test++) {
		gpu.ComputeMeansOld(d_out, d_in, m, n, N);
	}
	gpu.queue.finish();
	end = std::chrono::high_resolution_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000  << std::endl;


}




TEST_CASE( "Tedfst EdgeThresfdsfsdfshold.", "[del3]" ) {

	float threshold = 1;
	int window_size = 1;
	
	m = 43657;
	n = 1536;
	uni = std::uniform_int_distribution<int>(0, 255);
	std::vector<uint8_t> rand(m * n);
	std::vector<uint8_t> mask(m * n);
	std::vector<uint8_t> mads(m);
	std::vector<uint8_t> thresholds(m * n);
	for (auto& v: mads) { 
		v = round(uni(rng)); 
	}
	for (auto& v: thresholds) { 
		v = round(uni(rng)); 
	}

	for (auto& v: rand) { 
		v = round(uni(rng)); 
	}
	uni = std::uniform_int_distribution<int>(0, 1);
	for (auto& v: mask) { 
		v = round(uni(rng)); 
	}

	

	cl::Buffer data = gpu.InitBuffer(CL_MEM_READ_WRITE , m * n  * sizeof(uint8_t));
	gpu.WriteToBuffer(rand.data(), data, m * n * sizeof(uint8_t));
	cl::Buffer gpu_mads = gpu.InitBuffer(CL_MEM_READ_WRITE , m * sizeof(uint8_t));
	gpu.WriteToBuffer(mads.data(), gpu_mads, m * sizeof(uint8_t));
	cl::Buffer gpu_mask = gpu.InitBuffer(CL_MEM_READ_WRITE , m * n * sizeof(uint8_t));
	gpu.WriteToBuffer(mask.data(), gpu_mask, m * n * sizeof(uint8_t));
	cl::Buffer gpu_mask_out = gpu.InitBuffer(CL_MEM_READ_WRITE , m * n * sizeof(uint8_t));
	cl::Buffer gpu_thresholds = gpu.InitBuffer(CL_MEM_READ_WRITE , m * sizeof(float));
	gpu.WriteToBuffer(thresholds.data(), gpu_thresholds, m * sizeof(uint8_t));
		
	for (int test = 0; test < 2; test++) {
		//gpu.EdgeThreshold(gpu_mask_out, data, gpu_mads, threshold, 1, m, n,n, 16, 16);
		//gpu.SumThreshold(gpu_mask_out, data, gpu_mask, gpu_thresholds, window_size, m, n, 1, 256);
	}
	
	gpu.queue.finish();
	gpu.queue.flush();

	//for (int i = 0; i < 8; i++) {
		//auto begin = std::chrono::high_resolution_clock::now();
		//for (int test = 0; test < 1000; test++) {
			//gpu.EdgeThreshold(gpu_mask_out, data, gpu_mask, gpu_mads, threshold, std::pow(2,i), m, n, 1, 256);
		//}
		//gpu.queue.finish();
		//auto end = std::chrono::high_resolution_clock::now();
		//std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000 << std::endl;
	//}
	//gpu.queue.flush();
	
	for (int w = 1; w <= 10; w++) {
	auto begin = std::chrono::high_resolution_clock::now();
	for (int test = 0; test < 1000; test++) {
		//gpu.SumThreshold(gpu_mask_out, data, gpu_mask, gpu_thresholds, w, m, n, 1, 256);
		//for (int window_size = 1; window_size <= w; window_size++) {
			//gpu.SumThreshold(gpu_mask_out, data, gpu_mask, gpu_thresholds, window_size, m, n, 1, 256);
		gpu.EdgeThreshold(gpu_mask_out, data, gpu_mads, threshold, w, m, n,n, 1, 256);
		//}
	}
	gpu.queue.finish();
	auto end = std::chrono::high_resolution_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000  << std::endl;
	}


}



//TEST_CASE( "Test EdgeThreshold.", "[edge_threshold], [rfi]" ) {

	//float threshold = 1;
	//uint8_t median;
	//uint8_t value;
	//uint8_t window_stat;
	
	//for (int test = 0; test < 10; test++) {
		//InitExperiment(1000, 1000, 0, 256);
		
		//// Compute MADs
		//std::vector<uint8_t> mads(m);
		//std::vector<uint8_t> mask(m * n, 0);
		//std::vector<uint8_t> temp(vec);
		//for (int i = 0; i < m; i++) {
			//std::nth_element(temp.begin() + i * n, temp.begin() + i * n + n/2, temp.begin() + i * n + n);
			//median = temp[i * n + n/2];
			//std::transform(temp.begin() + i * n, temp.begin() + i * n + n, temp.begin() + i * n, [median](uint8_t x) -> uint8_t { return std::abs(x - median); });
			//std::nth_element(temp.begin() + i * n, temp.begin() + i * n + n/2, temp.begin() + i * n + n);
			//mads[i] = temp[i * n + n/2];
		//}
	
		//for (int i = 0; i < m; i++) {
			//for (int j = 1; j < n - 1; j++) {
				//window_stat = vec[i * n + j];
				//value = std::min(std::abs(window_stat - vec[i * n + j - 1]), std::abs(window_stat - vec[i * n + j + 1]));
				//if (std::abs(value / (1.4826 * mads[i])) > threshold) {
					//mask[i * n + j]	= 1;
				//}
	
			//}
		//}

		//cl::Buffer gpu_mads = gpu.InitBuffer(CL_MEM_READ_WRITE , m * sizeof(uint8_t));
		//cl::Buffer gpu_mask = gpu.InitBuffer(CL_MEM_READ_WRITE , m * n * sizeof(uint8_t));
		//gpu.WriteToBuffer(mads.data(), gpu_mads, m * sizeof(uint8_t));
		//gpu.EdgeThreshold(gpu_mask, gpu_mads, d_in, threshold, m, n, 12, 12);

		//gpu.ReadFromBuffer(results.data(), gpu_mask, n * m * sizeof(uint8_t));
	
		//CHECK_VEC_EQUAL(mask, results);

	//}

//}



//TEST_CASE( "MAD Performance.", "[del]" ) {

	//cl_ulong size;
	//clGetDeviceInfo(gpu.devices[0](), CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &size, 0);
	//cout << size << endl;

	//n = 1536;
	//m = 43657;
	 ////= 1536;
	//uni = std::uniform_int_distribution<int>(0, 255);
	//std::vector<uint8_t> rand(m * n);
	

	//cl::Buffer data = gpu.InitBuffer(CL_MEM_READ_WRITE , m * n  * sizeof(uint8_t));
	//gpu.WriteToBuffer(rand.data(), data, m * n * sizeof(uint8_t));
	//cl::Buffer medians = gpu.InitBuffer(CL_MEM_READ_WRITE , m * sizeof(uint8_t));
	//cl::Buffer mads = gpu.InitBuffer(CL_MEM_READ_WRITE , m * sizeof(uint8_t));
		
	//for (int test = 0; test < 5; test++) {
		//gpu.ComputeMads(mads, medians, data, m, n, 32, 1);
	//}
	
	//gpu.queue.finish();
	//gpu.queue.flush();

	//auto begin = std::chrono::high_resolution_clock::now();
	//for (int test = 0; test < 1000; test++) {
		//gpu.ComputeMads(mads, medians, data, m, n, 16, 8);
	//}
	//gpu.queue.finish();
	//auto end = std::chrono::high_resolution_clock::now();
	//std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000  << std::endl;
	//begin = std::chrono::high_resolution_clock::now();
	//for (int test = 0; test < 1000; test++) {
		//gpu.ComputeMads(mads, medians, data, m, n, 16, 16);
	//}
	//gpu.queue.finish();
	//end = std::chrono::high_resolution_clock::now();
	//std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000  << std::endl;
	//begin = std::chrono::high_resolution_clock::now();
	//for (int test = 0; test < 1000; test++) {
		//gpu.ComputeMads(mads, medians, data, m, n, 16, 32);
	//}
	//gpu.queue.finish();
	//end = std::chrono::high_resolution_clock::now();
	//std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000  << std::endl;
	//begin = std::chrono::high_resolution_clock::now();
	//for (int test = 0; test < 1000; test++) {
		//gpu.ComputeMads(mads, medians, data, m, n, 16, 64);
	//}
	//gpu.queue.finish();
	//end = std::chrono::high_resolution_clock::now();
	//std::cout << std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count() / 1000  << std::endl;






//}



