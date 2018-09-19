/*****************************************************************************
 AUTHOR:
 Jedda Boyle

 Notes:
 *****************************************************************************/



#include "filterbank.h"

#define GET_HEADER_FIELD(data_field) { \
    std::string s(#data_field); \
    std::string key = s.substr(s.find(".") + 1, std::string::npos ); \
    ReadFromHeader(key, typeid(data_field).name(), &data_field); \
}

template<typename T>
FilterBank<T>::FilterBank(const std::string& file_name, const FilterBankHeader& header) {
	out_file_stream.open(file_name, std::ios::binary | std::ios::out);
	WriteHeader (header);
	out_file_stream.close();
	Init(file_name);
}

template<typename T>
FilterBank<T>::FilterBank(const std::string& file_name) {
	Init(file_name);
}

template<typename T>
FilterBank<T>::~FilterBank() {
	out_file_stream.close();
	in_file_stream.close();
}

template<typename T>
void FilterBank<T>::Init(const std::string& file_name) {
	in_file_stream.open(file_name, std::ios::binary | std::ios::ate);
    out_file_stream.open(file_name, std::ios::binary | std::ios::ate| std::ios::app);
    // Compute the number of bytes in the header by searching for HEADER_END.
    std::string header_end_identifier("HEADER_END");
    std::string next_string(header_end_identifier.size(), '.');
    size_t pos = 0;
    while (next_string != header_end_identifier) {
        in_file_stream.seekg(pos++, std::ios::beg);
        in_file_stream.read(&next_string[0], next_string.size());
    }
    nbytes_header = in_file_stream.tellg();

    // Read header data into a vector.
    header_data.resize(nbytes_header);
    in_file_stream.seekg(0, std::ios::beg);
    in_file_stream.read(header_data.data(), nbytes_header);

    // Extract information from header.
    GET_HEADER_FIELD(header.pulsarcentric);
    GET_HEADER_FIELD(header.telescope_id);
    GET_HEADER_FIELD(header.barycentric);
    GET_HEADER_FIELD(header.source_name);
    GET_HEADER_FIELD(header.rawdatafile);
    GET_HEADER_FIELD(header.machine_id);
    GET_HEADER_FIELD(header.data_type);
    GET_HEADER_FIELD(header.fchannel);
    GET_HEADER_FIELD(header.nsamples);
    GET_HEADER_FIELD(header.az_start);
    GET_HEADER_FIELD(header.za_start);
    GET_HEADER_FIELD(header.src_raj);
    GET_HEADER_FIELD(header.src_dej);
    GET_HEADER_FIELD(header.nbeams);
    GET_HEADER_FIELD(header.tstart);
    GET_HEADER_FIELD(header.nchans);
    GET_HEADER_FIELD(header.period);
    GET_HEADER_FIELD(header.tsamp);
    GET_HEADER_FIELD(header.nbits);
    GET_HEADER_FIELD(header.ibeam);
    GET_HEADER_FIELD(header.refdm);
    GET_HEADER_FIELD(header.fch1);
    GET_HEADER_FIELD(header.foff);
    GET_HEADER_FIELD(header.nifs);

    // Compute meta data about filterbank file.
	in_file_stream.seekg(0, std::ios_base::end);
	nbytes_data = ((size_t) in_file_stream.tellg() - nbytes_header);
    nbytes_per_spectrum = header.nchans * (header.nbits / 8);
    nbins = nbytes_data / nbytes_per_spectrum;

    // Check that
	if (sizeof(T) != (header.nbits / 8)) {
		throw std::runtime_error("sizeof(T) != nbits / 8");
	}
}

template<typename T>
bool FilterBank<T>::ReadInSpectraBlock(std::vector<T>& spectra) {
    if (current_bin > nbins) {
        return true;
    }
    GetSpectra(spectra, current_bin, current_bin + nbins_per_block);
    current_bin += std::min(nbins_per_block, nbins - current_bin);
    return false;


}

template<typename T>
void FilterBank<T>::GetSpectra (std::vector<T>& spectra, const size_t start_bin, const size_t end_bin) {

	// TODO Check if we are trying to read invalid bins from filterbank file.
    size_t nbytes_to_read = nbytes_per_spectrum * (std::min(nbins, end_bin) - start_bin);
    size_t event_size = nbytes_to_read / sizeof(T);
    size_t start_byte = nbytes_header + (start_bin * nbytes_per_spectrum);

    spectra.resize(event_size);
	in_file_stream.seekg(start_byte, std::ios_base::beg);
    in_file_stream.read((char*) spectra.data(), nbytes_to_read);

}

template<typename T>
void FilterBank<T>::AppendSpectra(std::vector<T>& spectra) {
    size_t bytes_to_write = spectra.size() * sizeof(uint8_t);
    out_file_stream.write((char*) spectra.data(), bytes_to_write);
	nbins += spectra.size() / header.nchans;

}

template<typename T>
void FilterBank<T>::ReadFromHeader(const std::string key, const std::string type, void* data) {
    // Find the location of the data associated with key in the header.
    auto loc = search(header_data.begin(), header_data.end(), key.begin(), key.end()) + key.size();

    if (loc == header_data.end() + key.size()) {
        return;
    }
    if (type == "i") {
        std::memcpy(data, &*loc, 4);
    }
    else if (type == "d") {
        std::memcpy(data, &*loc, 8);

    }
    else { // Read in a string.
        int str_len;
		std::memcpy(&str_len, &*loc, 4);

		std::string *sp = static_cast<std::string*>(data);
        sp->resize(str_len);
        std::memcpy((void*) sp->data(), &*(loc + 4), str_len);
    }
}

template<typename T>
void FilterBank<T>::WriteRawString (std::string string){
	const char* c_str = string.c_str();
	size_t len = strlen(c_str);
	out_file_stream.write((char*) &len, sizeof(int));
	out_file_stream.write((char*) c_str, sizeof(char) * len);
}

template<typename T>
void FilterBank<T>::WriteString (std::string name, std::string value) {
	WriteRawString(name);
	WriteRawString(value);
}

template<typename T>
void FilterBank<T>::WriteHeader (const FilterBankHeader& header) {
	WriteRawString("HEADER_START");

	WriteString("rawdatafile", header.rawdatafile);
	WriteString("source_name", header.source_name);

	WriteNumeral("pulsarcentric", header.pulsarcentric);
	WriteNumeral("telescope_id", header.telescope_id);
	WriteNumeral("barycentric", header.barycentric);
	WriteNumeral("machine_id", header.machine_id);
	WriteNumeral("data_type", header.data_type);
	WriteNumeral("az_start", header.az_start);
	WriteNumeral("za_start", header.za_start);
	WriteNumeral("src_raj", header.src_raj);
	WriteNumeral("src_dej", header.src_dej);
	WriteNumeral("nchans", header.nchans);
	WriteNumeral("nbeams", header.nbeams);
	WriteNumeral("tstart", header.tstart);
	WriteNumeral("ibeam", header.ibeam);
	WriteNumeral("tsamp", header.tsamp);
	WriteNumeral("nbits", header.nbits);
	WriteNumeral("nifs", header.nifs);
	WriteNumeral("fch1", header.fch1);
	WriteNumeral("foff", header.foff);

	WriteRawString("HEADER_END");

}

template class FilterBank<uint8_t>;
