/******************************************
 author      : Zhizhong Pan
 E-mail      : zhizhop@g.clemson.edu
 *******************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>

//import OpenCL header
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "RGU.h"

#define LWS 256

//Globl Variables declare
static unsigned int GWS;
static double* vector1;       // The first input vector
static double* vector2;       // The secaond input vector

// OpenCL globals
cl_platform_id platform;
cl_context context;
cl_device_id *device;
cl_command_queue command;
cl_kernel multiKernel, reduceKernel;
cl_program program;
cl_mem inputVector1, inputVector2, output[2];


void
initCL()
{
    int err;
    size_t contxtSize, kernelSize;
    char *kernelSource;
    unsigned int gpuDevCount;
    
    err = RGUGetPlatformID(&platform);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &gpuDevCount);
    device = new cl_device_id[gpuDevCount];
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, gpuDevCount, device, NULL);
    
    cl_context_properties props[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0};
    context = clCreateContext(props, 1, &device[0], NULL, NULL, &err);
    command = clCreateCommandQueue(context,device[0],0,&err);
    
    kernelSource = RGULoadProgSource("kernel.cl","", &kernelSize);
    program = clCreateProgramWithSource(context,1, (const char **)&kernelSource, NULL, NULL);
    clBuildProgram(program,0,NULL,NULL,NULL,NULL);
    multiKernel = clCreateKernel(program,"multi",NULL);
    reduceKernel = clCreateKernel(program, "reduce", NULL);
    
}


void
setBuffers()
{
    inputVector1 = clCreateBuffer(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                GWS * sizeof(double),vector1,NULL);
    inputVector2 = clCreateBuffer(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                GWS * sizeof(double),vector2,NULL);


    output[0] = clCreateBuffer(context,CL_MEM_READ_WRITE, (GWS) * sizeof(double),NULL,NULL);
    output[1] = clCreateBuffer(context,CL_MEM_READ_WRITE, (GWS) * sizeof(double),NULL,NULL);
}


static double*
readFile(char* filename){
    unsigned int length;
    double* vector;
    FILE *filePtr;
    
    filePtr = fopen(filename, "r");
    if (filePtr != NULL){
        fread(&length, sizeof(unsigned int), 1, filePtr);    
        // make the init vector size be an integer multiple of LWS
        if (length % LWS != 0)
            GWS = LWS - length % LWS + length;
        vector = new double[GWS];
        fread(vector, sizeof(double), length , filePtr);
        
        // make sure the padding elements are zero
        for (int i  = length; i < GWS; i++)
            vector[i] = 0.0;
    }else{
        fprintf(stderr, "Can't open file %s \n", filename);
        exit(-1);
    }
    fclose(filePtr);
    return vector;
}


static double
dotProd(double* vector1, double* vector2)
{
    initCL();
    setBuffers();
    
    unsigned int from = 0, to = 1;
    unsigned int modVal;
    double result[1];
    double result2[GWS];
    size_t workItemCount[1] = {GWS};    // store global work size
    size_t localItemCout[1] = {LWS};
    
    clSetKernelArg(multiKernel,0,sizeof(cl_mem),(void *)&inputVector1);
    clSetKernelArg(multiKernel,1,sizeof(cl_mem),(void *)&inputVector2);
    clSetKernelArg(multiKernel,2,sizeof(cl_mem),(void *)output[0]);
    
    
    clEnqueueNDRangeKernel(command, multiKernel,1,NULL,workItemCount,localItemCout,0,NULL,NULL);

    while (workItemCount[0]  > 1) {

        modVal = workItemCount[0] % localItemCout[0]; 
       if (modVal != 0)
            workItemCount[0] += localItemCout[0] - modVal;

        clSetKernelArg(reduceKernel,0,sizeof(cl_mem),(void *)output[from]);
        clSetKernelArg(reduceKernel,1,sizeof(cl_mem),(void *)output[to]);
        
        // do ping-pong
        from = 1 - from;
        to = 1 -to;

        printf("=================================  before ======================================");
        printf("--------------------------------------- from ---------------------------------\n");
        clEnqueueReadBuffer(command,output[from],CL_TRUE,0,workItemCount[0]*sizeof(double),result2,0,NULL,NULL);
        for (int i = 0; i < workItemCount[0]; i++)
            printf("%f, ", result2[i]);
        printf("--------------------------------------- to ---------------------------------\n");
        clEnqueueReadBuffer(command, output[to],CL_TRUE,0,workItemCount[0]*sizeof(double),result2,0,NULL,NULL);
        for (int i = 0; i < workItemCount[0]; i++)
            printf("%f, ", result2[i]);

        clEnqueueNDRangeKernel(command,reduceKernel,1,NULL, workItemCount ,localItemCout,0,NULL,NULL);

        printf("=================================  after ======================================");
        clEnqueueReadBuffer(command,output[from],CL_TRUE,0,workItemCount[0]*sizeof(double),result2,0,NULL,NULL);
        for (int i = 0; i < workItemCount[0]; i++)
            printf("%f, ", result2[i]);
        printf("--------------------------------------- to ---------------------------------\n");
        clEnqueueReadBuffer(command, output[to],CL_TRUE,0,workItemCount[0]*sizeof(double),result2,0,NULL,NULL);
        for (int i = 0; i < workItemCount[0]; i++)
            printf("%f, ", result2[i]);


        workItemCount[0] = ceil( workItemCount[0]  / double (localItemCout[0]));
    }

    clEnqueueReadBuffer(command,output[from],CL_TRUE,0,1*sizeof(double),result,0,NULL,NULL);
        
    return result[0];
    
    
}

void
cleanup(int signo)
{
    int i;
    
    // clean device
    clReleaseProgram(program);
    clReleaseContext(context);
    clReleaseKernel(multiKernel);
    clReleaseKernel(reduceKernel);
    clReleaseCommandQueue(command);
    clReleaseMemObject(inputVector1);
    clReleaseMemObject(inputVector2);
    clReleaseMemObject(output[0]);
    clReleaseMemObject(output[1]);
    exit(0);
}


int
main(int argc, char * argv[])
{
    if (argc < 3) {
        fprintf(stderr, "Error! Usage: ./bigdot file file \n");
        exit(-1);
    }
    
    signal(SIGUSR1, cleanup);
    vector1 = readFile(argv[1]);
    vector2 = readFile(argv[2]);
    double result = dotProd(vector1, vector2);
    for (int i = 0; i < GWS; i++){
//	printf("%f ,", vector1[i]);
    }
    
    printf("Result: %f\n", result);
    cleanup(SIGUSR1);
    return 0;
}
