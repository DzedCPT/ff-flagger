/*****************************************************************************
 Jedda Boyle

 CONTAINS:
 OpenCL code defining the GPU kernels used for RFI mitigation.

 NOTES:

  *****************************************************************************/


kernel
void mask(global float *d_out, const global float *d_in, const global float* mask, float mask_value, uint m, uint n) {
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
void transpose(global float *d_out, const global float *d_in, uint m, uint n) {
	uint i = get_global_id(0);
	uint j = get_global_id(1);
	if (i < m && j < n) {
		d_out[j * m + i] = d_in[i * n + j];
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
void edge_threshold(global float *mask, global float* mads, global float *d_in, float threshold, uint m, uint n) {
	int i = get_global_id(0);
	int j = get_global_id(1) + 1;

	if (i >= m || j >= n - 1) { 
		return;
	}

	float window_stat = d_in[i * n + j];
	float value = fmin(fabs(window_stat - d_in[i * n + j - 1]), fabs(window_stat - d_in[i * n + j + 1]));
	if (mads[i] != 0 && fabs(value / mads[i]) > threshold) {
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
void mad_rows(global float *mads, global float *medians, global float *d_in, uint m, uint n) {
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
	uint median;
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
		uint cc = fabs(d_in[i * n + j] - median);
		xx[cc] += 1;
	}

	float MAD = 1;
	count = 0;
	for (int k = 0; k < 256; k++) {
		count += xx[k];
		if (count > n / 2) {
			MAD = 1.4826 * k;
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

