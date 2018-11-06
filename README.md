# FFlagger


-i,--input <string> [REQUIRED] path to input filterbank file.
                              
-o,--output <string> [REQUIRED] output filterbank file.
                              path to filterbank file written to.

-s,--seconds <float>=INF number of seconds to process, by default whole file is processed.

-m,--mode <int> = 1  [OPTIONS = 0, 1, 2, 3] 
  * mode 0: No RFI mitigation.
  * mode 1: Flag time samples with an outlier mean intensity.
  * mode 2: Edge thresholding.
  * mode 3: Edge thresholding then flag time samples with an outlier mean intensity.
  
--num_samples <int>=43657 number of sample per rfi event.

-n,--num_iteratations <int>=1 number of times to loop over the data.

--mad_threshold <float>=3.5 threshold used for edge thresholding.

--std_threshold <float>=2.5 threshold for flagging time samples.

--rfi_mode <int>=2 [OPTIONS = 1, 2] 
  * option 1: Replace RFI with zeros.
  * option 2: Replace RFI with median of the frequeny channel.
