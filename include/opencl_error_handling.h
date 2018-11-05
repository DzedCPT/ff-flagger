#ifndef OPENCL_ERROR_HANDLING
#define OPENCL_ERROR_HANDLING

#include <CL/cl.hpp>

#define CHECK_CL(cl_status) \
error_code = cl_status; \
if (error_code != CL_SUCCESS) { \
	fprintf(stderr, "OpenCL Error %s (%d)\n", GetErrorString(error_code), error_code); \
	fprintf(stderr, "The error occured in %s at line %d.\n", __FILE__, __LINE__); \
	exit(1); \
}

// https://stackoverflow.com/questions/24326432/convenient-way-to-show-opencl-error-codes
extern const char *GetErrorString(cl_int error);
		
#endif
