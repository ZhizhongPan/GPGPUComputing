#pragma OPENCL EXTENSION cl_khr_fp64 : enable

__kernel void
multi(__global double* voctor1,
      __global double* voctor2,
      __global double* result)
{
    size_t i = get_global_id(0);
    
    result[i] = voctor1[i] * voctor2[i];
    
}

__kernel void
reduce(__global double* input, __global double* output)
{
    __local double sdata[LWS];
    unsigned int tid = get_local_id(0);
    unsigned int i = get_global_id(0);
    unsigned int groups = get_num_groups(0);
    unsigned int s;
    
    if (i >= groups) output[i] = 0.0;
    
    sdata[tid] = input[i];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    
    for(s = LWS/2; s > 0; s >>= 1){
        if (tid < s) sdata[tid] += sdata[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (tid == 0) output[get_group_id(0)] = sdata[0];
}
