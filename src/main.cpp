#include "CLI11.hpp"
#include <iostream>
#include <string>
#include "filterbank.h"
#include "timing.h"
#include "device.h"
#include <CL/cl.hpp>

#include <sys/time.h>
typedef unsigned long long timestamp_t;
timestamp_t get_timestamp () {
	  struct timeval now;
	  gettimeofday (&now, NULL);
	  return  now.tv_usec + (timestamp_t)now.tv_sec * 1000000;
}


void ProcessFilterBank(FilterBank<uint8_t>& in_fil_file, FilterBank<uint8_t>& out_fil_file, float threshold, float row_threshold, const size_t nbins, float total_time = 0) {
	std::vector<uint8_t> spectra;
	//size_t m = nbins;
	//size_t n = in_fil_file.header.nchans;
	RFIPipeline rfi_pipeline(in_fil_file.header.nchans, nbins);
	cl::Buffer uint_buffer   = rfi_pipeline.InitBuffer(CL_MEM_READ_WRITE, nbins * in_fil_file.header.nchans * sizeof(uint8_t));
	cl::Buffer uint_buffer_T = rfi_pipeline.InitBuffer(CL_MEM_READ_WRITE, nbins * in_fil_file.header.nchans * sizeof(uint8_t));
	//cl::Buffer mask   = gpu.InitBuffer(CL_MEM_READ_WRITE, m * n * sizeof(uint8_t));
	//cl::Buffer mask_T = gpu.InitBuffer(CL_MEM_READ_WRITE, m * n * sizeof(uint8_t));

	//cl::Buffer time_mads = gpu.InitBuffer(CL_MEM_READ_WRITE, m * sizeof(uint8_t));
	//cl::Buffer time_medians = gpu.InitBuffer(CL_MEM_READ_WRITE, m * sizeof(uint8_t));
	//cl::Buffer freq_mads = gpu.InitBuffer(CL_MEM_READ_WRITE, n * sizeof(uint8_t));
	//cl::Buffer freq_medians = gpu.InitBuffer(CL_MEM_READ_WRITE , n * sizeof(uint8_t));

    INIT_TIMER(timer);
    INIT_MARK(mark);
	std::cout << "hello" << std::endl;

    in_fil_file.nbins_per_block = nbins;
    total_time = (total_time != 0) ? total_time : in_fil_file.nbins * in_fil_file.header.tsamp;
    while(in_fil_file.tellg() < total_time) {

		// Read in the data.
		in_fil_file.ReadInSpectraBlock(spectra);
		rfi_pipeline.WriteToBuffer(spectra.data(), uint_buffer_T, spectra.size() * sizeof(uint8_t));
		rfi_pipeline.Transpose(uint_buffer, uint_buffer_T, nbins, in_fil_file.header.nchans, 12, 12);

		MARK_TIME(mark);
		rfi_pipeline.BasicFlagger(uint_buffer);
		//timestamp_t t1 = get_timestamp();
		//gpu.queue.enqueueFillBuffer(mask, 0, 0, n * m * sizeof(uint8_t));

		//gpu.ComputeMads(time_mads, time_medians, uint_buffer, m, n, 500);
		//gpu.EdgeThreshold(mask, time_mads, uint_buffer, threshold, m, n, 12, 12);
		//gpu.OutlierDetection(time_medians, m, 5, threshold, 1000);
		//gpu.ConstantRowMask(mask, time_medians, m, n, 500);

		//gpu.Transpose(uint_buffer_T, uint_buffer, m, n, 12, 12);
		//gpu.Transpose(mask_T, mask, m, n, 12, 12);

		//gpu.ComputeMads(freq_mads, freq_medians, uint_buffer_T, n, m, 500);
		//gpu.EdgeThreshold(mask_T, freq_mads, uint_buffer_T, threshold, n, m, 12, 12);

		//gpu.ReplaceRFI(uint_buffer_T, uint_buffer_T, mask_T, freq_medians, m, n, 12, 12);

		//gpu.Transpose(uint_buffer, uint_buffer_T, n, m, 12, 12);
		

		ADD_TIME_SINCE_MARK(timer, mark);
		rfi_pipeline.Transpose(uint_buffer_T, uint_buffer, in_fil_file.header.nchans, nbins,12, 12);
		rfi_pipeline.ReadFromBuffer(spectra.data(), uint_buffer_T, spectra.size() * sizeof(uint8_t));

		out_fil_file.AppendSpectra(spectra);
        std::cout << "\rProgress "
                  << std::min(in_fil_file.tellg() / total_time, (float) 1.0) * 100
                  << " % " << std::flush;
    }
    std::cout << std::endl;

    PRINT_TIMER(timer);



}


int main(int argc, char *argv[]) {
	CLI::App app("AmberAlert - RFI mitigation\nFlag time bins and frequency channels with extreme mean values.");
	std::string in_file_path;
	app.add_option("-i,--input", in_file_path, "path to filterbank file read from.")->required();

	std::string out_file_path;
	app.add_option("-o,--output", out_file_path, "path to filterbank file written to.")->required();

	float time_seconds = 0;
	app.add_option("-s,--seconds", time_seconds, "# of seconds that will be processed.", true);

	float sigma_threshold = 3.5;
	app.add_option("-st,--sigma_threshold", threshold, "# of standard deviations from the mean required for the band to be flagged.", true);

	int mode = 1;
	app.add_option("-m,--mode", threshold, "# of standard deviations from the mean required for the band to be flagged.", true);
    //float row_threshold = 0.3;
	//app.add_option("-r,--row_threshold", row_threshold, "", true);

	int n = 43657;
	app.add_option("-n,--num_samples", n, "# of time bins per event.", true);

	CLI11_PARSE(app, argc, argv);

	FilterBank<uint8_t> in_fil_file(in_file_path);
	FilterBank<uint8_t> out_fil_file(out_file_path, in_fil_file.header);

	ProcessFilterBank(in_fil_file, out_fil_file, threshold, row_threshold, n, time_seconds);

}
