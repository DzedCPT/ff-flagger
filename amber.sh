#!/bin/bash

#if ! [ "$hostname" == "arts022" ]; then
#    echo "Error: can only be executed on arts022"
#    exit 1
#fi

conf_dir=$HOME/software/ARTS-obs/amber_conf/
#file=/data2/CB500DM.fil
#file=/data2/snr_tests_liam/dm250_10frbs20180419-145517.fil
#file=/data2/snr_tests_liam/dm125_50frbs20180417-092302.fil
#file=/data2/output/snr_tests_liam/20180430/dm250.0_nfrb50_20180430-0840.fil
file=$1
hdr=455
nbatch=$2
snrmin=6.5

amber -opencl_platform 0 -opencl_device 0 -device_name ARTS_step1 -padding_file $conf_dir/padding.conf -zapped_channels $conf_dir/zapped_channels.conf -integration_steps $conf_dir/integration_steps.conf -integration_file $conf_dir/integration.conf -subband_dedispersion -dedispersion_step_one_file $conf_dir/dedispersion_stepone.conf -dedispersion_step_two_file $conf_dir/dedispersion_steptwo.conf -snr_file $conf_dir/snr.conf -output amber_step1 -subbands 32 -dms 32 -dm_first 0 -dm_step .1 -subbanding_dms 128 -subbanding_dm_first 0 -subbanding_dm_step 3.2 -threshold $snrmin -sigproc -header $hdr -data $file -batches $nbatch -channel_bandwidth .1953125 -min_freq 1250.09765625 -channels 1536 -samples 25000 -sampling_time 4.096e-05 -stream &

amber -opencl_platform 0 -opencl_device 1 -device_name ARTS_step2 -padding_file $conf_dir/padding.conf -zapped_channels $conf_dir/zapped_channels.conf -integration_steps $conf_dir/integration_steps.conf -integration_file $conf_dir/integration.conf -subband_dedispersion -dedispersion_step_one_file $conf_dir/dedispersion_stepone.conf -dedispersion_step_two_file $conf_dir/dedispersion_steptwo.conf -snr_file $conf_dir/snr.conf -output amber_step2 -subbands 32 -dms 32 -dm_first 0 -dm_step .2 -subbanding_dms 128 -subbanding_dm_first 409.6 -subbanding_dm_step 6.4 -threshold $snrmin -sigproc -header $hdr -data $file -batches $nbatch -channel_bandwidth .1953125 -min_freq 1250.09765625 -channels 1536 -samples 25000 -sampling_time 4.096e-05 -stream &

amber -opencl_platform 0 -opencl_device 2 -device_name ARTS_step3 -padding_file $conf_dir/padding.conf -zapped_channels $conf_dir/zapped_channels.conf -integration_steps $conf_dir/integration_steps.conf -integration_file $conf_dir/integration.conf -subband_dedispersion -dedispersion_step_one_file $conf_dir/dedispersion_stepone.conf -dedispersion_step_two_file $conf_dir/dedispersion_steptwo.conf -snr_file $conf_dir/snr.conf -output amber_step3 -subbands 32 -dms 32 -dm_first 0 -dm_step .5 -subbanding_dms 128 -subbanding_dm_first 1228.8 -subbanding_dm_step 16 -threshold $snrmin -sigproc -header $hdr -data $file -batches $nbatch -channel_bandwidth .1953125 -min_freq 1250.09765625 -channels 1536 -samples 25000 -sampling_time 4.096e-05 -stream &

wait

#cat amber_step?.trigger > amber.trigger
#cat amber_step?.trigger > $1.trigger
cat amber_step?.trigger > $3.trigger
