#include "CLI11.hpp"
#include <iostream>
#include <string>
#include "filterbank.h"
#include "timing.h"
#include "device.h"
#include <CL/cl.hpp>

void ProcessFilterBank(FilterBank<uint8_t>& in_fil_file, FilterBank<uint8_t>& out_fil_file, float threshold, const size_t nbins, float total_time = 0) {
	std::vector<uint8_t> spectra;
	size_t m = nbins;
	size_t n = in_fil_file.header.nchans;
	GPUEnviroment gpu;
	cl::Buffer uint_buffer = gpu.InitBuffer(CL_MEM_READ_WRITE , m * n * sizeof(uint8_t));
	cl::Buffer float_buffer = gpu.InitBuffer(CL_MEM_READ_WRITE , m * n  * sizeof(float));
	cl::Buffer mask = gpu.InitBuffer(CL_MEM_READ_WRITE , m * n * sizeof(float));
	cl::Buffer mads = gpu.InitBuffer(CL_MEM_READ_WRITE , m * sizeof(float));

    INIT_TIMER(timer);
    INIT_MARK(mark);


    in_fil_file.nbins_per_block = nbins;
    total_time = (total_time != 0) ? total_time : in_fil_file.nbins * in_fil_file.header.tsamp;
    while(in_fil_file.tellg() < total_time) {
		in_fil_file.ReadInSpectraBlock(spectra);
		gpu.WriteToBuffer(spectra.data(), uint_buffer, spectra.size() * sizeof(uint8_t));
		gpu.Upcast(float_buffer, uint_buffer, spectra.size(), 500);

		MARK_TIME(mark);
		gpu.queue.enqueueFillBuffer(mask, 0, 0, in_fil_file.header.nchans * nbins * sizeof(float));

		gpu.MADRows(mads, float_buffer, m, n, 500);
		gpu.EdgeThreshold(mask, mads, float_buffer, 2, m, n, 12, 12);

		gpu.Mask(float_buffer, float_buffer, mask, m, n, 12, 12);
		gpu.Downcast(uint_buffer, float_buffer, spectra.size(), 500);
		gpu.ReadFromBuffer(spectra.data(), uint_buffer, spectra.size() * sizeof(uint8_t));

		ADD_TIME_SINCE_MARK(timer, mark);


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

	float threshold = 3.5;
	app.add_option("-t,--threshold", threshold, "# of standard deviations from the mean required for the band to be flagged.", true);

	int n = 5000;
	app.add_option("-n,--num_bins", n, "# of time bins per event.", true);

	CLI11_PARSE(app, argc, argv);

	FilterBank<uint8_t> in_fil_file(in_file_path);
	FilterBank<uint8_t> out_fil_file(out_file_path, in_fil_file.header);

	ProcessFilterBank(in_fil_file, out_fil_file, threshold, n, time_seconds);

}
