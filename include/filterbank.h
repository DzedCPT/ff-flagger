/*****************************************************************************
 AUTHOR:
 Jedda Boyle

 CONTAINS:
 FilterBank class used to read and write filterbank files.

 NOTES:
 This implementation of a filterbank reader/writer adheres to the sigproc
 standard with some exceptions.
 1. npuls header field is omitted.
 2. FREQUENCY_START header field is omitted.
 3. FREQUENCY_END header field is omitted
 4. Does not handle folded-filterbank files

 This filterbank class is compatible with the AMBER filterbank
 reader and the filterbank writer used at the westerbork telecope.
 https://github.com/AA-ALERT/dadafilterbank/blob/master/filterbank.c

 *****************************************************************************/

#ifndef FILTERBANK_H
#define FILTERBANK_H

#include <iostream>
#include <fstream>
#include <vector>
#include <iostream>
#include <typeinfo>
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <cmath>


// Filterbank header from page 4 of http://sigproc.sourceforge.net/sigproc.pdf,
// retreived 2017-05-31 but *EXCLUDES* FREQUENCY_START and FREQUENCY_END.
struct FilterBankHeader {
    std::string rawdatafile;   // rawdatafile: the name of the original data file
    std::string source_name;   // source name: the name of the source being observed by the telescope
    int nchans;                // nchans: number of filterbank channels
    int nifs;                  // ifs: number of seperate IF channels
    int telescope_id;          // telescope_id : 0=fake data; 1=Arecibo; 2=Ooty
    int machine_id;            // machine_id: 0=fake; 1=PSPM; 2=WAPP; 3=Ooty
    int data_type;             // data_type: 1=filterbank; 2=time series
    int barycentric;           // barycentric: equals 1 if data are barycentric or 0 otherwise
    int pulsarcentric;         // pulsarcentric: equals 1 if data are pulsarcentric or 0 otherwise
    int nbits;                 // nbits: number of bits per time sample
    int nsamples;              // nsamples: number of time samples in the data file (rarely used any more)
    int nbeams;                // NOT DOCUMENTED BUT IN USE IN THE SIGPROC CODE
    int ibeam;                 // NOT DOCUMENTED BUT IN USE IN THE SIGPROC CODE
    double fch1;               // fch1: centre frequency (MHz) of first filterbank channel
    double foff;               // foff: filterbank channel bandwidth (MHz)
    double fchannel;           // fchannel: frequency channel value (MHz)
    double refdm;              // refdm: reference dispersion measure (cm−3 pc)
    double period;             // period: folding period (s)
    double az_start;           // az_start: telescope azimuth at start of scan (degrees)
    double za_start;           // za_start: telescope zenith angle at start of scan (degrees)
    double src_raj;            // src_raj: right ascension (J2000) of source (hhmmss.s)
    double src_dej;            // src_dej: declination (J2000) of source (ddmmss.s)
    double tstart;             // tstart: time stamp (MJD) of first sample
    double tsamp;              // tsamp: time interval between samples (s)

};



template<typename T>
class FilterBank {
public:

    FilterBankHeader header;

    std::ofstream out_file_stream;
    std::ifstream in_file_stream;

    size_t nbytes_header;
    size_t nbytes_data;
    size_t nbytes_per_spectrum;
    size_t nbins;
    size_t nbins_per_block = 10000;
    size_t current_bin = 0;

    std::vector<char> header_data;

    /**
    Initialse new filterbank object using the meta data in header.
    */
    FilterBank (const std::string& file_name, const FilterBankHeader& header);

    /**
    Open already existing filterbank file.
    */
    FilterBank (const std::string& file_name);

    ~FilterBank();

    float tellg() {
        return header.tsamp * current_bin;
    }

    bool ReadInSpectraBlock(std::vector<T>& spectra);

    void GetSpectra(std::vector<T>& spectra, const size_t start_bin, const size_t end_bin);

    void AppendSpectra(std::vector<T>& spectra);

private:
    void Init (const std::string& file_name);

    void ReadFromHeader(const std::string key, const std::string type, void* data);

    void WriteRawString (std::string string);

    void WriteString (std::string name, std::string value);

    template<typename U>
    void WriteNumeral (std::string name, U value) {
        WriteRawString(name);
        out_file_stream.write((char*) &value, sizeof(U));
    }

    void WriteHeader (const FilterBankHeader& header);

};




#endif
