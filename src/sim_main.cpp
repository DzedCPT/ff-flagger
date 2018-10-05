#include "CLI11.hpp"
#include <iostream>
#include <string>
#include "filterbank.h"
#include "timing.h"
#include "device.h"
#include <CL/cl.hpp>

void ProcessFilterBank(FilterBank<uint8_t>& in_fil_file, FilterBank<uint8_t>& out_fil_file, float threshold, float row_threshold, const size_t nbins, float total_time = 0) {
	std::vector<uint8_t> spectra;
	size_t m = nbins;
	size_t n = in_fil_file.header.nchans;
	GPUEnviroment gpu;
	cl::Buffer uint_buffer = gpu.InitBuffer(CL_MEM_READ_WRITE , m * n * sizeof(uint8_t));
	cl::Buffer uint_buffer_T = gpu.InitBuffer(CL_MEM_READ_WRITE , m * n * sizeof(uint8_t));
	cl::Buffer bin_medians = gpu.InitBuffer(CL_MEM_READ_WRITE , m * sizeof(uint8_t));
	cl::Buffer freq_medians = gpu.InitBuffer(CL_MEM_READ_WRITE , n * sizeof(uint8_t));

    INIT_TIMER(timer);
    INIT_MARK(mark);


    in_fil_file.nbins_per_block = nbins;
    total_time = (total_time != 0) ? total_time : in_fil_file.nbins * in_fil_file.header.tsamp;
    while(in_fil_file.tellg() < total_time) {
		in_fil_file.ReadInSpectraBlock(spectra);

		gpu.WriteToBuffer(spectra.data(), uint_buffer, spectra.size() * sizeof(uint8_t));
		//float mean = gpu.Reduce(float_buffer_T, 100, m * n, 1000) / (m * n);

		MARK_TIME(mark);
		gpu.Transpose(uint_buffer_T, uint_buffer, m, n, 25, 25);
		gpu.ComputeMedians(freq_medians, uint_buffer_T, n, m, 500);
		//gpu.queue.enqueueFillBuffer(row_mask, 0, 0, m * sizeof(uint8_t));

		gpu.ComputeMedians(bin_medians, uint_buffer, m, n, 500);
		gpu.OutlierDetection(bin_medians, m, 5, threshold, 1000);
		gpu.MaskRows(uint_buffer, bin_medians, freq_medians, m, n, 500);

		ADD_TIME_SINCE_MARK(timer, mark);

		gpu.ReadFromBuffer(spectra.data(), uint_buffer, spectra.size() * sizeof(uint8_t));

        out_fil_file.AppendSpectra(spectra);
        std::cout << "\rProgress "
                  << std::min(in_fil_file.tellg() / total_time, (float) 1.0) * 100
                  << " % " << std::flush;
    }
    std::cout << std::endl;

    PRINT_TIMER(timer);
	gpu.PrintKernelTimers();



}


int main(int argc, char *argv[]) {
	CLI::App app("AmberAlert - RFI mitigation\nFlag time bins and frequency channels with extreme mean values.");
	std::string in_file_path;
	app.add_option("-i,--input", in_file_path, "path to filterbank file read from.")->required();

	std::string out_file_path;
	app.add_option("-o,--output", out_file_path, "path to filterbank file written to.")->required();

	float time_seconds = 0;
	app.add_option("-s,--seconds", time_seconds, "# of seconds that will be processed.", true);

	float threshold = 3.5;
	app.add_option("-t,--threshold", threshold, "# of standard deviations from the mean required for the band to be flagged.", true);

    float row_threshold = 0.3;
	app.add_option("-r,--row_threshold", row_threshold, "", true);

	int n = 5000;
	app.add_option("-n,--num_bins", n, "# of time bins per event.", true);

	CLI11_PARSE(app, argc, argv);

	FilterBank<uint8_t> in_fil_file(in_file_path);
	FilterBank<uint8_t> out_fil_file(out_file_path, in_fil_file.header);

	ProcessFilterBank(in_fil_file, out_fil_file, threshold, row_threshold, n, time_seconds);

}
