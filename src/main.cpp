#include "CLI11.hpp"
#include <iostream>
#include <string>
#include <assert.h>
#include "filterbank.h"
#include "device.h"
#include <CL/cl.hpp>

#include <sys/time.h>

//struct Params {
	//int mode;
	//float time;
	//int n_samples;
	//int max_window_size;
	//int n;
	//float mad_threshold;
	//float std_threshold;

//};

//void ProcessFilterBank(FilterBank<uint8_t>& in_fil_file, 
					   //FilterBank<uint8_t>& out_fil_file, 
					   //float threshold, 
					   //float row_threshold, 
					   //const size_t nbins, 
					   //float total_time = 0) 

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

		//MARK_TIME(mark);

		begin = std::chrono::high_resolution_clock::now();
		rfi_pipeline.Flag(uint_buffer);
		rfi_pipeline.queue.finish();
		end = std::chrono::high_resolution_clock::now();
		rfi_timer += std::chrono::duration_cast<std::chrono::milliseconds>(end-begin).count();

		//ADD_TIME_SINCE_MARK(timer, mark);
		rfi_pipeline.Transpose(uint_buffer_T, uint_buffer, in_fil_file.header.nchans, params.n_samples, 12, 12);
		rfi_pipeline.ReadFromBuffer(spectra.data(), uint_buffer_T, spectra.size() * sizeof(uint8_t));

		out_fil_file.AppendSpectra(spectra);
		std::cout << "\rProgress "
				  << std::min(in_fil_file.tellg() / total_time, (float) 1.0) * 100
				  << " % " << std::flush;
	}
	//std::cout << std::endl;
	std::cout << "\rRFI mitigation took " << rfi_timer << " milliseconds to process " << total_time << " seconds of data." << std::endl;
	rfi_pipeline.PrintTimers();


}


int main (int argc, char *argv[]) 
{
	CLI::App app("AmberAlert - RFI mitigation\nFlag time bins and frequency channels with extreme mean values.");
	std::string in_file_path;
	app.add_option("-i,--input", in_file_path, "path to filterbank file read from.")->required();

	std::string out_file_path;
	app.add_option("-o,--output", out_file_path, "path to filterbank file written to.")->required();

	RFIPipeline::Params params;
	params.mode = 1;
	app.add_option("-m,--mode", params.mode, "# of standard deviations from the mean required for the band to be flagged.", true);
	
	float time = 0;
	app.add_option("-s,--seconds", time, "# of seconds that will be processed.", true);

	params.n_samples = 43657;
	app.add_option("--num_samples", params.n_samples, "# of time bins per event.", true);
	params.n_padded_samples = params.n_samples;

	params.n_iter = 1;
	app.add_option("-n,--num_iteratations", params.n_iter, "# of time bins per event.", true);

	params.mad_threshold = 3.5;
	app.add_option("--mad_threshold", params.mad_threshold, "# of standard deviations from the mean required for the band to be flagged.", true);

	params.std_threshold = 2.5;
	app.add_option("--std_threshold", params.std_threshold, "# of standard deviations from the mean required for the band to be flagged.", true);

	int rfi_mode = 2;
	app.add_option("--rfi_mode", rfi_mode, "# of standard deviations from the mean required for the band to be flagged.", true);

	CLI11_PARSE(app, argc, argv);

	FilterBank<uint8_t> in_fil_file(in_file_path);
	FilterBank<uint8_t> out_fil_file(out_file_path, in_fil_file.header);
	
	assert(0 <= rfi_mode && rfi_mode <= 2);
	params.n_channels = in_fil_file.header.nchans;
	params.rfi_replace_mode = (rfi_mode == 1 ? RFIPipeline::RFIReplaceMode::ZEROS : RFIPipeline::RFIReplaceMode::MEDIANS);
    
	ProcessFilterBank(in_fil_file, out_fil_file, params, time);

}
