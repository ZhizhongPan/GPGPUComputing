
__kernel void
multi(__global double* voctor1,
      __global double* voctor2,
      __global double* result)
{
    size_t i = get_global_id(0);
    
    result[i] = voctor1[i] * voctor2[i];
    
}

__kernel void
dotReducer(__global double* input,
           __global double* output)
{
    __local double localResult[LWS];
    unsigned int localId = get_local_id(0);
    unsigned int globalID = get_global_id(0);
    unsigned int s;

    output[globalID] = 0.0;
    localResult[localId] = input[globalID];
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for(s = LWS/2; s > 0; s >>= 1){
        if (localId < s) localResult[localId] += localResult[localId + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (localId == 0) output[get_group_id(0)] = localResult[0];
    
}
