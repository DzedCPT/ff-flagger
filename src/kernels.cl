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
void compute_deviation(global float *d_out, global float *d_in, float mean, int n) {
	uint gid = get_global_id(0);
	if (gid < n) {
		/*d_out[gid] =  pow(mean - d_in[gid], 2) / (n - 1);*/
		/*d_out[gid] =  pow(mean - d_in[gid], 2);*/
		d_out[gid] =  pow(d_in[gid], 2);
	}

}
/*kernel*/
/*void transpose(global uchar *d_out, const global uchar *d_in, int m, int n, local uchar *tile) {*/
	/*int tile_dim = get_local_size(1)*/
	/*uint group_x = get_group_id(0);*/
	/*uint group_y = get_group_id(1);*/
	/*uint x = group_x * tile_dim + get_local_id(0);*/
	/*uint y = get_global_id(1);*/
	/*for (uint i = 0; i < tile_dim; i += get_local_size(0)) {*/
		/*tile[get_local_id(1) * tile_dim + (get_local_id(0)+i)] = d_in[(x + i) * n + y];*/
	/*}*/
	/*barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);*/
	
	/*group_x = get_group_id(1);*/
	/*group_y = get_group_id(0);*/
	/*x = group_x * tile_dim + get_local_id(0);*/
	/*y = group_y * tile_dim + get_local_id(1);*/
		/*for (uint i = 0; i < tile_dim; i += get_local_size(0)) {*/
		/*if (x + i >= n || y >=m ) {*/
			/*return;*/
		/*}*/

		/*d_out[(x + i) * m + y] = tile[(get_local_id(0) + i) * tile_dim  + get_local_id(1)];*/
	/*}*/


/*}*/

kernel
void transpose(global uchar *d_out, global uchar *d_in, int m, int n, local uchar *ldata) {
	int gid_x = get_global_id(0);
	int gid_y = get_global_id(1);
	int tid_x = get_local_id(0);
	int tid_y = get_local_id(1);
	/*int n_threads_x = get_local_size(1);*/
	int n_threads_x = 16;

	if (gid_x < n && gid_y < m) {
		ldata[tid_y * (n_threads_x + 1) + tid_x] = d_in[gid_y * n + gid_x];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	gid_x = get_group_id(1) * n_threads_x + get_local_id(0);
	gid_y = get_group_id(0) * n_threads_x + get_local_id(1);
	if (gid_x < m && gid_y < n) {
		d_out[gid_y * n + gid_x] = ldata[tid_x * (n_threads_x + 1) + tid_y];
	}

}

/*kernel*/
/*void transpose(global uchar *d_out, const global uchar *d_in, int m, int n, local uchar *ldata) {*/
	/*int gid_x = get_global_id(0);*/
	/*int gid_y = get_global_id(1);*/
	/*if (gid_x < m && gid_y < n) {*/
		/*d_out[gid_y * m + gid_x] = d_in[gid_x * n + gid_y];*/
	/*}*/
/*}*/


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
void sum_threshold(global uchar *mask_out, global uchar *d_in, global uchar *mask, global uchar *thresholds, int window_size, int m, int n, local float *ldata, local float *lmask, local float *lthresholds) {
	int gid_x = get_global_id(0);
	int gid_y = get_global_id(1);
	/*int tid_x = get_local_id(0);*/
	/*int tid_y = get_local_id(1);*/
	/*int tile_width = get_local_size(1) + window_size;*/
	/*if (gid_x < m && gid_y < n) {*/
		/*ldata[tid_x * tile_width + tid_y] = d_in[gid_x * n + gid_y];*/
		/*lmask[tid_x * tile_width + tid_y] = mask[gid_x * n + gid_y];*/
	/*}*/

	/*if (tid_y == 0) {*/
		/*lthresholds[tid_x] = thresholds[gid_x];*/
	/*}*/

	/*int d = get_local_size(0) - tid_y - 1;  */
	/*if (d <= window_size && gid_y + window_size < n) {*/
		/*ldata[tid_x * tile_width + tid_y + window_size] = d_in[gid_x * n + gid_y + window_size];*/
		/*lmask[tid_x * tile_width + tid_y + window_size] = mask[gid_x * n + gid_y + window_size];*/
	/*}*/

	/*barrier(CLK_LOCAL_MEM_FENCE);*/


	if (gid_x >= m || gid_y >= n - window_size + 1) {
		return;
	
	}


	float window_sum = 0;
	int count = 0;
	for (int i = 0; i < window_size; i++) {
		if (mask[gid_x * n + gid_y + i] != 1) {
		/*if (lmask[tid_x * tile_width + tid_y + i] != 1) {*/
			count += 1;
			window_sum += d_in[gid_x * n + gid_y + i];
			/*window_sum += ldata[tid_x * tile_width + tid_y + i];*/
		}
	}

	if (window_sum > thresholds[gid_x] * count) {
	/*if (window_sum > lthresholds[tid_x] * count) {*/
		for (int i = 0; i < window_size; i++) {
			mask_out[gid_x * n + gid_y + i] = 1;
		}
	}


}
/*kernel */
/*void edge_threshold(global uchar *mask_out, global uchar* d_in, global uchar *mask, global uchar *mads, float threshold, int window_size, int m, int n, local float *ldata, local float *lmads, local float *lmask) {*/
	/*int gid_x = get_global_id(0);*/
	/*int gid_y = get_global_id(1) + 1;*/
	/*int tid_x = get_local_id(0);*/
	/*int tid_y = get_local_id(1) + 1;*/
	/*int tile_width = 1 + get_local_size(1) + window_size;*/
	/*int tid = tid_x * tile_width + tid_y;*/
	/*int gid = gid_x * n + gid_y;*/

	
	/*if (gid_x < m && gid_y < n) {*/
		/*[>[>[>ldata[tid_x * tile_width + tid_y] = d_in[gid_x * n + gid_y];<]<]<]*/
		/*ldata[tid] = d_in[gid];*/
		/*[>[>[>[>[>lmask[tid_x * tile_width + tid_y] = mask[gid_x * n + gid_y];<]<]<]<]<]*/
	/*}*/

	/*if (tid_y == 1) {*/
		/*lmads[tid_x] = mads[gid_x];*/
		/*[>[>[>ldata[tid_x * tile_width + tid_y - 1] = d_in[gid_x * n + gid_y - 1];<]<]<]*/
		/*ldata[tid - 1] = d_in[gid - 1];*/
	/*}*/
	/*int d = get_local_size(0) - tid_y - 1;  */
	/*if (d <= window_size && gid_y + window_size < n) {*/
		/*[>[>[>ldata[tid_x * tile_width + tid_y + window_size] = d_in[gid_x * n + gid_y + window_size];<]<]<]*/
		/*ldata[tid + window_size] = d_in[gid + window_size];*/
		/*[>[>[>[>[>lmask[tid_x * tile_width + tid_y + window_size] = mask[gid_x * n + gid_y + window_size];<]<]<]<]<]*/
	/*}*/

	/*barrier(CLK_LOCAL_MEM_FENCE);*/

	/*[>if (gid_x >= m || gid_y >= n - window_size || mask[gid_x * n + gid_y - 1] == 1 || mask[gid_x * n + gid_y + window_size] == 1) { <]*/
	/*[>if (gid_x >= m || gid_y >= n - window_size || lmask[tid_x * tile_width + tid_y - 1] == 1 || lmask[tid_x * tile_width + tid_y + window_size] == 1) { <]*/
	/*for (int w = 1; w <= window_size; w++) {*/
	/*if (gid_x >= m || gid_y >= n - window_size) { */
		/*return;*/
	/*}*/

	/*float window_stat = 0;*/
	/*for (int i = 0; i < window_size; i++) {*/
		/*[>window_stat += ldata[tid_x * tile_width + tid_y + i];<]*/
		/*window_stat += ldata[tid + i];*/
		/*[>window_stat += d_in[gid_x * n + gid_y + i];<]*/
	/*}*/
	/*window_stat /= window_size;*/
	/*float value = (float) min(fabs(window_stat - ldata[tid - 1]), fabs(window_stat - ldata[tid + window_size]));*/
	/*[>float value = (float) min(fabs(window_stat - d_in[gid_x * n + gid_y - 1]), fabs(window_stat - d_in[gid_x * n + gid_y + window_size]));<]*/
	/*[>if (value / mads[gid_x] > (1.4826 * threshold)) {<]*/
	/*if (value / lmads[tid_x] > (1.4826 * threshold)) {*/
		/*for (int i = 0; i < window_size; i++) {*/
			/*mask_out[gid + i] = 1;*/
		/*}*/

	/*}*/
	/*}*/
/*}*/
kernel
void edge_threshold(global uchar *mask_out, global uchar* d_in, global uchar *mask, global uchar *mads, float threshold, int window_size, int m, int n, local float *ldata, local float *lmads, local float *lmask) {
	int gid_x = get_global_id(0);
	int gid_y = get_global_id(1) + 1;
	int tid_x = get_local_id(0);
	int tid_y = get_local_id(1) + 1;
	int tile_width = 1 + get_local_size(1) + window_size;
	int tid = tid_x * tile_width + tid_y;
	int gid = gid_x * n + gid_y;

	
	if (gid_x < m && gid_y < n) {
		/*[>[>[>ldata[tid_x * tile_width + tid_y] = d_in[gid_x * n + gid_y];<]<]<]*/
		ldata[tid] = d_in[gid];
		/*[>[>[>[>[>lmask[tid_x * tile_width + tid_y] = mask[gid_x * n + gid_y];<]<]<]<]<]*/
	}

	if (tid_y == 1) {
		lmads[tid_x] = mads[gid_x];
		/*[>[>[>ldata[tid_x * tile_width + tid_y - 1] = d_in[gid_x * n + gid_y - 1];<]<]<]*/
		ldata[tid - 1] = d_in[gid - 1];
	}
	int d = get_local_size(0) - tid_y - 1;  
	if (d <= window_size && gid_y + window_size < n) {
		/*[>[>[>ldata[tid_x * tile_width + tid_y + window_size] = d_in[gid_x * n + gid_y + window_size];<]<]<]*/
		ldata[tid + window_size] = d_in[gid + window_size];
		/*[>[>[>[>[>lmask[tid_x * tile_width + tid_y + window_size] = mask[gid_x * n + gid_y + window_size];<]<]<]<]<]*/
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	/*if (gid_x >= m || gid_y >= n - window_size || mask[gid_x * n + gid_y - 1] == 1 || mask[gid_x * n + gid_y + window_size] == 1) { */
	/*if (gid_x >= m || gid_y >= n - window_size || lmask[tid_x * tile_width + tid_y - 1] == 1 || lmask[tid_x * tile_width + tid_y + window_size] == 1) { */
	for (int w = 1; w <= window_size; w++) {
		if (gid_x >= m || gid_y >= n - w) { 
			continue;
		}

		float window_stat = 0;
		for (int i = 0; i < w; i++) {
			for (int j = 0; j < w; j++) {
			
				window_stat += ldata[tid_x * tile_width + tid_y + j];
			
			}
			/*window_stat += ldata[tid + i];*/
			/*window_stat += d_in[gid_x * n + gid_y + i];*/
		}
		window_stat /= w;
		float value = (float) min(fabs(window_stat - ldata[tid - 1]), fabs(window_stat - ldata[tid + w]));
		/*float value = (float) min(fabs(window_stat - d_in[gid_x * n + gid_y - 1]), fabs(window_stat - d_in[gid_x * n + gid_y + window_size]));*/
		/*if (value / mads[gid_x] > (1.4826 * threshold)) {*/
		if (value / lmads[tid_x] > (1.4826 * threshold)) {
			for (int i = 0; i < w; i++) {
				mask_out[gid + i] = 1;
			}

		}
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

kernel void compute_means(global float *d_out, global uchar *d_in, int m, int n, local volatile int *ldata) {
	int gid = get_global_id(0);
	int tid = get_local_id(1);
	int lid = get_local_id(0) * get_local_size(1) + tid; // index into local memory/
	if (gid >= m) return;

	ldata[lid] = 0;
	for (int i = tid; i < n; i += get_local_size(1)) {
		ldata[lid] += d_in[gid * n + i];
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	if (tid < 64) {
		ldata[lid] += ldata[lid + 64];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	if (tid < 32) {
		ldata[lid] += ldata[lid + 32]; 
		barrier(CLK_LOCAL_MEM_FENCE);
		ldata[lid] += ldata[lid + 16];
		barrier(CLK_LOCAL_MEM_FENCE);
		ldata[lid] += ldata[lid +  8];
		barrier(CLK_LOCAL_MEM_FENCE);
		ldata[lid] += ldata[lid +  4];
		barrier(CLK_LOCAL_MEM_FENCE);
		ldata[lid] += ldata[lid +  2];
		barrier(CLK_LOCAL_MEM_FENCE);
		ldata[lid] += ldata[lid +  1];
	}

	if (tid == 0) d_out[gid] = (float) ldata[lid] / n;

	
	

}

kernel void reduce(global float *d_out, global float *d_in, int n, local volatile float *ldata) {
	int tid = get_local_id(0);
	int gid = 256 * get_group_id(0) + tid; // Each work group reduces 256 values.

	// Read data into shared memory and do first reduce operation.
	ldata[tid] = d_in[gid] + (gid + 128 < n ? d_in[gid + 128] : 0);
	barrier(CLK_LOCAL_MEM_FENCE);

	// Do reduce across work group.
	if (tid < 64) {
		ldata[tid] += ldata[tid + 64];
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	// Do reduce across warp.
	if (tid < 32) {
		ldata[tid] += ldata[tid + 32]; 
		ldata[tid] += ldata[tid + 16];
		ldata[tid] += ldata[tid +  8];
		ldata[tid] += ldata[tid +  4];
		ldata[tid] += ldata[tid +  2];
		ldata[tid] += ldata[tid +  1];
	}

	// Write result.
	if (tid == 0) d_out[get_group_id(0)] = ldata[0];
}


