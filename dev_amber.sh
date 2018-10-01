#!/bin/bash

conf_dir=/home/oostrum/compare/confs

#file=/data2/output/dm1500.0_nfrb50_20180430-0840.fil
#file=/data2/output/snr_tests_liam/20180823/dm10.0_nfrb5000_10240_sec_20180823-0916.fil
#file=/data2/output/snr_tests_liam/20180823/dm10.0_nfrb5000_clean.fil
file=$1
hdr=455
nbatch=$2

#file=/data2/output/snr_tests_liam/20180816/dm500.0_nfrb8_16_sec_20180816-1238.fil
#file=/data2/output/snr_tests_liam/20180816/dm351.0_nfrb50_102_sec_20180816-1043.fil
#hdr=455
#nbatch=100

snrmin=6

amber -opencl_platform 0 -opencl_device 1 -device_name ARTS_step2 -sync -print -snr_momad -padding_file $conf_dir/padding.conf -zapped_channels $conf_dir/zapped_channels.conf -integration_steps $conf_dir/integration_steps.conf -integration_file $conf_dir/integration.conf -subband_dedispersion -dedispersion_stepone_file $conf_dir/dedispersion_stepone.conf -dedispersion_steptwo_file $conf_dir/dedispersion_steptwo.conf -max_file $conf_dir/max.conf -mom_stepone_file $conf_dir/mom_stepone.conf -mom_steptwo_file $conf_dir/mom_steptwo.conf -momad_file $conf_dir/momad.conf -output $3 -subbands 32 -dms 32 -dm_first 0 -dm_step 0.2 -subbanding_dms 128 -subbanding_dm_first 5.0 -subbanding_dm_step 6.4 -threshold $snrmin -sigproc -stream -header $hdr -data $file -batches $nbatch -channel_bandwidth .1953125 -min_freq 1250.09765625 -channels 1536 -samples 25000 -sampling_time 4.096e-05

