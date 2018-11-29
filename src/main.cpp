#include "CLI11.hpp"
#include <iostream>
#include <string>
#include <assert.h>
#include "filterbank.h"
#include "device.h"
#include <CL/cl.hpp>

#include <sys/time.h>

void ProcessFilterBank (FilterBank<uint8_t>& in_fil_file, 
		               FilterBank<uint8_t>& out_fil_file, 
					   const RFIPipeline::Params& params,
					   const float& time)
{
	std::vector<uint8_t> spectra(params.n_channels * params.n_samples);
	RFIPipeline rfi_pipeline(params);

	auto begin = std::chrono::high_resolution_clock::now();
	auto end = std::chrono::high_resolution_clock::now();
	double rfi_timer = 0.0;

	cl::Buffer uint_buffer   = rfi_pipeline.InitBuffer(CL_MEM_READ_WRITE, params.n_samples * in_fil_file.header.nchans * sizeof(uint8_t));
	cl::Buffer uint_buffer_T = rfi_pipeline.InitBuffer(CL_MEM_READ_WRITE, params.n_samples * in_fil_file.header.nchans * sizeof(uint8_t));

	in_fil_file.nbins_per_block = params.n_samples;
	float total_time = (time != 0) ? time : in_fil_file.nbins * in_fil_file.header.tsamp;
	while(in_fil_file.tellg() < total_time) {

		// Read in the data.
		in_fil_file.ReadInSpectraBlock(spectra);
		rfi_pipeline.WriteToBuffer(spectra.data(), uint_buffer_T, spectra.size() * sizeof(uint8_t));
		rfi_pipeline.Transpose(uint_buffer, uint_buffer_T, params.n_samples, in_fil_file.header.nchans, 12, 12);

		rfi_pipeline.queue.finish();
		begin = std::chrono::high_resolution_clock::now();

		rfi_pipeline.Flag(uint_buffer);

		rfi_pipeline.queue.finish();
		end = std::chrono::high_resolution_clock::now();
		rfi_timer += std::chrono::duration_cast<std::chrono::microseconds>(end-begin).count();

		rfi_pipeline.Transpose(uint_buffer_T, uint_buffer, in_fil_file.header.nchans, params.n_samples, 12, 12);
		rfi_pipeline.ReadFromBuffer(spectra.data(), uint_buffer_T, spectra.size() * sizeof(uint8_t));

		out_fil_file.AppendSpectra(spectra);
		std::cout << "\rProgress "
				  << std::min(in_fil_file.tellg() / total_time, (float) 1.0) * 100
				  << " % " << std::flush;
	}
	std::cout << "\rRFI mitigation took " << rfi_timer << " microseconds to process " << total_time << " seconds of data." << std::endl;
	std::cout << rfi_pipeline.time << std::endl;
	rfi_pipeline.PrintTimers();


}


int main (int argc, char *argv[]) 
{
	CLI::App app("FF-Flagger RFI mitigation software.");

	// Read in command line arguments.

	// Required input and output files.
	std::string in_file_path;
	app.add_option("-i", in_file_path, "Path to input filterbank file.")->required();

	std::string out_file_path;
	app.add_option("-o", out_file_path, "Path to output filterbank file.")->required();

	// Read in RFI mitigation parameters.
	RFIPipeline::Params params;

	params.mode = 3;
	app.add_option("-m", params.mode, "RFI Mode, options are:\n\t\t\t\t1. Point edge-thresholding.\n\t\t\t\t2. Flag 0-DM RFI bursts.\n\t\t\t\t3. Full FF-Flagger pipeline", true);

	float time = 0;
	app.add_option("-s", time, "Number of seconds of data to process. Default is entire file.", true);

	params.n_iter = 5;
	app.add_option("-n", params.n_iter, "Number of 0 DM RFI removal loops.", true);

	params.max_window_size = 3;
	app.add_option("-w", params.max_window_size, "Maximum window size used for point edge-thresholding.", true);

	params.zero_dm_threshold = 2.5;
	app.add_option("-z", params.zero_dm_threshold, "Threshold for 0 DM RFI removal.", true);

	params.edge_threshold = 3.5;
	app.add_option("-p", params.edge_threshold, "Threshold used for edge-thresholding.", true);

	params.stat_freq = 1;
	app.add_option("-f", params.stat_freq, "Frequency for recomputing MADs and medians.", true);
	

	params.n_samples = 43657;
	app.add_option("--num_samples", params.n_samples, "Number of time samples per RFI mitigation window.", true);
	params.n_padded_samples = params.n_samples;

	CLI11_PARSE(app, argc, argv);

	FilterBank<uint8_t> in_fil_file(in_file_path);
	FilterBank<uint8_t> out_fil_file(out_file_path, in_fil_file.header);
	
	params.n_channels = in_fil_file.header.nchans;

	ProcessFilterBank(in_fil_file, out_fil_file, params, time);

}
