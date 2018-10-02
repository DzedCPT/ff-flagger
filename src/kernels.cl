/*****************************************************************************
 Jedda Boyle

 CONTAINS:
 OpenCL code defining the GPU kernels used for RFI mitigation.

 NOTES:

  *****************************************************************************/


kernel
void mask(global uchar *d_out, const global uchar *d_in, const global uchar* mask, uint mask_value, uint m, uint n) {
	uint i = get_global_id(0);
	uint j = get_global_id(1);
	if (i < m && j < n) {
		d_out[i * n + j] = mask[i *n + j] == 1 ? mask_value : d_in[i * n + j];
	}

}

kernel
void downcast(global uchar *d_out, const global float *d_in, uint len) {
	uint i = get_global_id(0);
	if (i < len) {
		d_out[i] = d_in[i];
	}
}

kernel
void transpose(global uchar *d_out, const global uchar *d_in, uint m, uint n) {
	uint i = get_global_id(0);
	uint j = get_global_id(1);
	if (i < m && j < n) {
		d_out[j * m + i] = d_in[i * n + j];
	}
}
kernel 
void grubb(global uchar *data, uint len, uint work_per_thread, float threshold, local float *local_mem, local float *pad) { 
	uint global_data_index = get_global_id(0) * work_per_thread;
	uint local_data_index = get_local_id(0) * work_per_thread;
	uint work_group_index = get_local_id(0);
	uint work_group_size = get_local_size(0);
	uint work_group_data_size = work_group_size * work_per_thread;

	if (global_data_index >= len) {
		return;
	}
	if (get_group_id(0) == get_num_groups(0) - 1 ) {
		work_group_data_size = len - (get_group_id(0) * work_group_size * work_per_thread );
		work_group_size = ceil((float) work_group_data_size / work_per_thread);
	
	}
	uint work = min(work_per_thread, len - global_data_index);

	float sum = 0;
	for (uint k = global_data_index, kk=local_data_index; k < global_data_index + work; k++, kk++) {
		local_mem[kk] = data[k];
		sum += data[k];
	}
	pad[get_local_id(0)] = sum;
	
	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	int stride = 1;
	while (stride < work_group_size) {
		if (work_group_index % (2 * stride) == 0 && work_group_index + stride < work_group_size) {
			pad[work_group_index] += pad[work_group_index + stride];
		}
		stride *= 2;	
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
		 
	}
	float mean = pad[0] / (work_group_data_size);

	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	float total_deviation = 0;
	for (uint kk=local_data_index; kk < local_data_index + work; kk++) {
		total_deviation += pow(local_mem[kk] - mean, 2) / (work_group_data_size - 1) ;
	}
	pad[get_local_id(0)] = total_deviation;

	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	stride = 1;
	while (stride < work_group_size) {
		if (work_group_index % (2 * stride) == 0 && work_group_index + stride < work_group_size) {
			pad[work_group_index] += pad[work_group_index + stride];
		}
		stride *= 2;	
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
		 
	}
	float std = sqrt(pad[0]);

	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	for (uint k = global_data_index; k < global_data_index + work; k++) {
		data[k] = (fabs(data[k] - mean) > threshold * std);
	}


}


/*kernel */
/*void grubb(global float *data, uint len, uint work_per_thread, float threshold, local float *local_mem, local float *pad) { */
	/*uint global_data_index = get_global_id(0) * work_per_thread;*/
	/*uint local_data_index = get_local_id(0) * work_per_thread;*/
	/*uint work_group_index = get_local_id(0);*/
	/*uint work_group_size = get_local_size(0);*/
	/*uint work_group_data_size = work_group_size * work_per_thread;*/

	/*if (global_data_index >= len) {*/
		/*return;*/
	/*}*/
	/*if (get_group_id(0) == get_num_groups(0) - 1 ) {*/
		/*work_group_data_size = len - (get_group_id(0) * work_group_size * work_per_thread );*/
		/*work_group_size = ceil((float) work_group_data_size / work_per_thread);*/
	
	/*}*/
	/*uint work = min(work_per_thread, len - global_data_index);*/

	/*float sum = 0;*/
	/*for (uint k = global_data_index, kk=local_data_index; k < global_data_index + work; k++, kk++) {*/
		/*local_mem[kk] = data[k];*/
		/*sum += data[k];*/
	/*}*/
	/*pad[get_local_id(0)] = sum;*/
	
	/*barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);*/

	/*int stride = 1;*/
	/*while (stride < work_group_size) {*/
		/*if (work_group_index % (2 * stride) == 0 && work_group_index + stride < work_group_size) {*/
			/*pad[work_group_index] += pad[work_group_index + stride];*/
		/*}*/
		/*stride *= 2;	*/
		/*barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);*/
		 
	/*}*/
	/*float mean = pad[0] / (work_group_data_size);*/

	/*barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);*/

	/*float total_deviation = 0;*/
	/*for (uint kk=local_data_index; kk < local_data_index + work; kk++) {*/
		/*total_deviation += pow(local_mem[kk] - mean, 2) / (work_group_data_size - 1) ;*/
	/*}*/
	/*pad[get_local_id(0)] = total_deviation;*/

	/*barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);*/

	/*stride = 1;*/
	/*while (stride < work_group_size) {*/
		/*if (work_group_index % (2 * stride) == 0 && work_group_index + stride < work_group_size) {*/
			/*pad[work_group_index] += pad[work_group_index + stride];*/
		/*}*/
		/*stride *= 2;	*/
		/*barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);*/
		 
	/*}*/
	/*float std = sqrt(pad[0]);*/

	/*barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);*/

	/*for (uint k = global_data_index; k < global_data_index + work; k++) {*/
		/*data[k] = (fabs(data[k] - mean) > threshold * std);*/
	/*}*/


/*}*/

/*void GPUEnviroment::MaskRows(const cl::Buffer& data, cl::Buffer& mask, uint8_t mask_value, size_t m, size_t n, size_t local_size) {*/
kernel 
void mask_rows(global uchar *data, global uchar *mask, uchar mask_value, uint m, uint n) {
	uint i = get_global_id(0);
	if (i < m && mask[i] == 1) {
		for (uint j = 0; j < n; j++) {
			data[i * n + j] = mask_value;
		}
	}
}



kernel 
void upcast(global float *d_out, const global uchar *d_in, uint len) {
	uint i = get_global_id(0);
	if (i < len) {
		d_out[i] = d_in[i];
	}
}

kernel 
void edge_threshold(global uchar *mask, global uchar* mads, global uchar *d_in, float threshold, uint m, uint n) {
	int i = get_global_id(0);
	int j = get_global_id(1) + 1;

	if (i >= m || j >= n - 1) { 
		return;
	}

	uchar window_stat = d_in[i * n + j];
	float value = (float) min(abs(window_stat - d_in[i * n + j - 1]), abs(window_stat - d_in[i * n + j + 1]));
	if (mads[i] != 0 && fabs(value / mads[i]) > (1.4826 * threshold)) {
		mask[i * n + j]	= 1;
	}
	

}
	
kernel 
void flag_rows(global float *mask, float row_sum_threshold, uint m, uint n) {
	int i = get_global_id(0);

	if (i >= m) { return; }

	// Count the number of masked cells in the rows.
	uint count = 0;
	for (int j = 0; j < n; j++) {
		count += mask[i * n + j];
	}
	
	// Mask the entire row if the number of masked cells is above some threshold.
	if ( count > row_sum_threshold) {
		for (int j = 0; j < n; j++) { mask[i * n + j] = 1; }
	}

}
	
kernel 
void mad_rows(global uchar *mads, global uchar *medians, global uchar *d_in, uint m, uint n) {
	int i = get_global_id(0);

	if (i >= m) { 
		return;
	}
	uint xx[256];
	for (int k = 0; k < 256; k++) {
		xx[k] = 0;
	}


	for (int j = 0; j < n; j++) {
		uint cc = d_in[i * n + j];
		xx[cc] += 1;
	}

	uint count = 0;
	uchar median;
	for (uint k = 0; k < 256; k++) {
		count += xx[k];
		if (count > n / 2) {
			median = k;
			medians[i] = median;
			break;
		
		}
	}
	for (int k = 0; k < 256; k++) {
		xx[k] = 0;
	}

	for (int j = 0; j < n; j++) {
		uint cc = abs(d_in[i * n + j] - median);
		xx[cc] += 1;
	}

	uchar MAD = 1;
	count = 0;
	for (int k = 0; k < 256; k++) {
		count += xx[k];
		if (count > n / 2) {
			MAD = k;
			break;
		}
	}

	mads[i] = MAD;
		
}


kernel 
void reduce(global float *in_data, global float *out_data, uint len, local float *local_mem) {
	uint i = get_global_id(0);
	uint local_i = get_local_id(0);
	if (i >= len) {
		local_mem[get_local_id(0)] = 0;
		return;	
	}
	local_mem[get_local_id(0)] = in_data[i];

	barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

	uint local_n = get_local_size(0);
	uint stride = 1;
	while (stride < local_n) {
		if (local_i % (2 * stride) == 0 && local_i + stride < local_n) {
			local_mem[local_i] += local_mem[local_i + stride];
		}
		stride *= 2;
		barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
	
	}

	if (local_i == 0) {
		out_data[get_group_id(0)] = local_mem[0];
	}

}

