/*****************************************************************************
 Jedda Boyle

 CONTAINS:
 OpenCL code defining the GPU kernels used for RFI mitigation.

 NOTES:

  *****************************************************************************/


kernel
void replace_rfi(global uchar *clean_data, global uchar *data, global uchar* rfi_mask, global uchar* replace_values, uint m, uint n) {
	uint i = get_global_id(0);
	uint j = get_global_id(1);
	if (i < m && j < n) {
		clean_data[i * n + j] = rfi_mask[i * n + j] == 1 ? replace_values[i] : data[i * n + j];
	}

}

kernel
void compute_deviation(global float *d_out, global float *d_in, float mean, uint len) {
	uint gid = get_global_id(0);
	if (gid < len) {
		d_out[gid] =  pow(mean - d_in[gid], 2) / (len - 1);
	}

}
/*kernel*/
/*void transpose(global uchar *d_out, const global uchar *d_in, uint tile_dim, uint m, uint n, local uchar *tile) {*/
	/*[>uint tile_dim = tile_dim_n;<]*/
	/*[>uint x = get_group_id(0) * tile_dim_m + get_local_id(0);<]*/
	/*[>uint x = get_global_id(0);<]*/
	/*[>uint y = get_group_id(1) * tile_dim_n + get_local_id(1);<]*/
	/*uint group_x = get_group_id(0);*/
	/*uint group_y = get_group_id(1);*/
	/*uint x = group_x * tile_dim + get_local_id(0);*/
	/*[>uint y = group_x * tile_dim + get_local_id(0);<]*/
	/*uint y = get_global_id(1);*/
	/*[>uint work_group_m = get_local_size(0);<]*/
	/*[>uint work_group_n = get_local_size(1);<]*/
	/*[>if (x >= m || y >=n ) {<]*/
		/*[>return;<]*/
	/*[>}<]*/
	/*for (uint i = 0; i < tile_dim; i += get_local_size(0)) {*/
		/*tile[get_local_id(1) * tile_dim + (get_local_id(0)+i)] = d_in[(x + i) * n + y];*/
	/*}*/
	/*barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);*/
	
	/*[>x = get_global_id(1);<]*/
	/*[>x = get_group_id(1) * tile_dim_n + get_local_id(0);<]*/
	/*[>y = get_group_id(0) * tile_dim_m + get_local_id(1);<]*/

	/*[>if (get_global_id(0) == 0 && get_global_id(1) == 0) {<]*/
	/*[>if (get_local_id(0) == 0 && get_local_id(1) == 0 && get_group_id(0) == get_num_groups(0) - 1) {<]*/
		/*[>d_out[x * n + y] = y;<]*/
		/*[>for (uint i = 0; i < tile_dim_m; i ++) {<]*/
			/*[>for (uint j = 0; j < tile_dim_m; j++) {<]*/
				/*[>d_out[i * n + j] = tile[i * tile_dim_n + j];<]*/

			/*[>}<]*/
		/*[>}<]*/
		/*[>uint group_x = get_group_id(0);<]*/
		/*[>uint group_y = get_group_id(1);<]*/
		/*[>x = group_x * tile_dim_m;<]*/
		/*[>y = group_y * tile_dim_m;<]*/
		/*[>d_out[y * m + x] = get_group_id(0) ;<]*/
		/*[>d_out[(y - 1) * m + x] = get_group_id(1);<]*/
	/*[>}<]*/
	/*group_x = get_group_id(1);*/
	/*group_y = get_group_id(0);*/
	/*x = group_x * tile_dim + get_local_id(0);*/
	/*y = group_y * tile_dim + get_local_id(1);*/
	/*[>y = get_global_id(1);<]*/
	/*[>if (x >= n || y >=m ) {<]*/
		/*[>return;<]*/
	/*[>}<]*/

	/*for (uint i = 0; i < tile_dim; i += get_local_size(0)) {*/
		/*if (x + i >= n || y >=m ) {*/
			/*return;*/
		/*}*/

		/*d_out[(x + i) * m + y] = tile[(get_local_id(0) + i) * tile_dim  + get_local_id(1)];*/
	/*}*/

	/*[>if (get_group_id(0) == 2 && get_group_id(1) == 0) {<]*/
		/*[>d_out[get_local_id(0) * m + get_local_id(1)] = 99 + get_local_id(0);<]*/

	/*[>}<]*/

/*}*/

/*void transpose(global uchar *d_out, const global uchar *d_in, uint m, uint n) {*/

kernel
void transpose(global uchar *d_out, const global uchar *d_in, uint m, uint n) {
	uint i = get_global_id(0);
	uint j = get_global_id(1);
	if (i < m && j < n) {
		d_out[j * m + i] = d_in[i * n + j];
	}
}


kernel
void detect_outliers(global float *d_out, global float *d_in, float mean, float std, float threshold, uint len) { 
	uint gid = get_global_id(0);
	if (gid < len) {
		d_out[gid] = (fabs(d_in[gid] - mean) > std * threshold);	
	}


}
/*kernel */
/*void detect_outliers(global uchar *data, uint len, uint work_per_thread, float threshold, local float *local_mem, local float *pad) { */
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
void mask_rows(global uchar *d_out, global float *mask, uint m, uint n) {
	uint gid_m = get_global_id(0);
	if (gid_m < m && mask[gid_m] == 1) {
		uint group_size_n = get_local_size(1);
		for (uint i = get_local_id(1); i < n; i += group_size_n) {
			d_out[gid_m * n + i] = 1;
		}
	}
}

kernel 
void constant_row_mask(global uchar *data, global uchar *mask, uint m, uint n) {
	uint i = get_global_id(0);
	if (i < m && mask[i] == 1) {
		for (uint j = 0; j < n; j++) {
			data[i * n + j] = 1;
		}
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
void compute_medians(global uchar *medians, global uchar *data, uint m, uint n) {
	int i = get_global_id(0);

	if (i >= m) { 
		return;
	}
	uint xx[256];
	for (int k = 0; k < 256; k++) {
		xx[k] = 0;
	}


	for (int j = 0; j < n; j++) {
		uint cc = data[i * n + j];
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
	
}


kernel 
void compute_mads(global uchar *mads, global uchar *medians, global uchar *data, uint m, uint n) {
	int i = get_global_id(0);

	if (i >= m) { 
		return;
	}
	uint xx[256];
	for (int k = 0; k < 256; k++) {
		xx[k] = 0;
	}


	for (int j = 0; j < n; j++) {
		uint cc = data[i * n + j];
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
		uint cc = abs(data[i * n + j] - median);
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

kernel void compute_means(__global float *d_out, __global uchar *d_in, uint m, uint n, __local int* sdata) {
	unsigned int gid = get_global_id(0);
	uint sdata_index = get_local_id(0) * get_local_size(1) + get_local_id(1);
	sdata[sdata_index] = 0;
	if (gid >= m) {
		return;
	}
	for (uint i = get_local_id(1); i < n; i += get_local_size(1)) {
		sdata[sdata_index] += d_in[gid * n + i];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	/*uint k = get_local_id(0) * get_local_size(1);*/
	if (get_local_id(1) == 0) {
		for (uint i = 1; i < get_local_size(1); i++) {
			sdata[sdata_index] += sdata[sdata_index + i];
		}	
		d_out[gid] = (float) sdata[sdata_index] / n;
	}

	
	

}

kernel void reduce(__global float *g_idata, __global float *g_odata, unsigned int n, __local volatile float* sdata) {
	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = get_local_id(0);
	unsigned int i = get_group_id(0)*(get_local_size(0)*2) + get_local_id(0);

	sdata[tid] = (i < n) ? g_idata[i] : 0;
	if (i + get_local_size(0) < n) 
		sdata[tid] += g_idata[i+get_local_size(0)];  

	barrier(CLK_LOCAL_MEM_FENCE);

	// do reduction in shared mem
	#pragma unroll 1
	for(unsigned int s = get_local_size(0)/2; s>32; s>>=1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	uint blockSize = get_local_size(0);
	if (tid < 32) {
		if (blockSize >=  64) { sdata[tid] += sdata[tid + 32]; }
		if (blockSize >=  32) { sdata[tid] += sdata[tid + 16]; }
		if (blockSize >=  16) { sdata[tid] += sdata[tid +  8]; }
		if (blockSize >=   8) { sdata[tid] += sdata[tid +  4]; }
		if (blockSize >=   4) { sdata[tid] += sdata[tid +  2]; }
		if (blockSize >=   2) { sdata[tid] += sdata[tid +  1]; }
	}

	// write result for this block to global mem 
	if (tid == 0) g_odata[get_group_id(0)] = sdata[0];
}

/*kernel */
/*void reduce(global float *in_data, global float *out_data, uint len, local float *local_mem) {*/
	/*uint i = get_global_id(0);*/
	/*uint local_i = get_local_id(0);*/
	/*if (i >= len) {*/
		/*local_mem[get_local_id(0)] = 0;*/
		/*return;	*/
	/*}*/
	/*local_mem[get_local_id(0)] = in_data[i];*/

	/*barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);*/

	/*uint local_n = get_local_size(0);*/
	/*uint stride = 1;*/
	/*while (stride < local_n) {*/
		/*if (local_i % (2 * stride) == 0 && local_i + stride < local_n) {*/
			/*local_mem[local_i] += local_mem[local_i + stride];*/
		/*}*/
		/*stride *= 2;*/
		/*barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);*/
	
	/*}*/

	/*if (local_i == 0) {*/
		/*out_data[get_group_id(0)] = local_mem[0];*/
	/*}*/

/*}*/

