/*****************************************************************************
 Jedda Boyle

 CONTAINS:
 OpenCL code defining the GPU kernels used for RFI mitigation.

 NOTES:

  *****************************************************************************/


kernel
void mask(global float *d_out, const global float *d_in, const global float* mask, uint m, uint n) {
	uint i = get_global_id(0);
	uint j = get_global_id(1);
	if (i < m && j < n) {
		d_out[i * n + j] = d_in[i * n + j] * (1 - mask[i * n + j]);
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
void mad_rows(global float *mads, global float *d_in, uint m, uint n) {
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
