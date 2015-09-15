/******************************************
 author      : Zhizhong Pan
 E-mail      : zhizhop@g.clemson.edu

 compile : g++ -O -I /usr/local/cuda/include bigdot.cpp -lOpenCL -lm -lXmu -o bigdot
*******************************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <unistd.h>
#include <string.h>
#include <signal.h>
#include <CL/cl.h>

#include "RGU.h"
#include "kernel.h"

//Global Variables declare
unsigned int GWS;
static double* vector1;       // The first input vector
static double* vector2;       // The secaond input vector

// cl global
cl_platform_id platform;
cl_context context;
cl_device_id *device;
cl_command_queue command;
cl_kernel multiKernel;
cl_kernel reduceKernel;
cl_program program;
cl_mem menVector1, menVector2, output[2];

void
initCL()
{
    int err;
    unsigned int gpudevcount;
    size_t kernelSize;
    char *kernelSource;
    const char *header;
    
    err = RGUGetPlatformID(&platform);
    err = clGetDeviceIDs(platform,CL_DEVICE_TYPE_GPU,0,NULL,&gpudevcount);
    device = new cl_device_id[gpudevcount];
    err = clGetDeviceIDs(platform,CL_DEVICE_TYPE_GPU,gpudevcount,device,NULL);
    cl_context_properties props[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platform, 0};
    context = clCreateContext(props,1,&device[0],NULL,NULL,&err);
    command = clCreateCommandQueue(context,device[0],0,&err);
    header = RGULoadProgSource("kernel.h", "", &kernelSize);
    kernelSource = RGULoadProgSource("kernel.cl",header, &kernelSize);
    program = clCreateProgramWithSource(context,1, (const char **)&kernelSource, &kernelSize, &err);
    err = clBuildProgram(program,0,NULL,NULL,NULL,NULL);

    free(kernelSource);

    multiKernel = clCreateKernel(program,"multi",NULL);
    reduceKernel = clCreateKernel(program, "dotReducer", NULL);
}


void
setBuffers()
{
    //Input vector in card memory
    menVector1 = clCreateBuffer(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                GWS * sizeof(double),vector1,NULL);
    menVector2 = clCreateBuffer(context,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,
                                GWS * sizeof(double),vector2,NULL);
    
    output[0] = clCreateBuffer(context,CL_MEM_READ_WRITE, GWS * sizeof(double),NULL,NULL);
    output[1] = clCreateBuffer(context,CL_MEM_READ_WRITE, GWS * sizeof(double),NULL,NULL);
}


static double
dotProd(double* vector1, double* vector2){
    initCL();
    setBuffers();

    size_t workItemCount[1] = {GWS};    // store global work size
    size_t localItemCout[1] = {256};       // store loacl work size
    double result[1];                       // final result.
    
    unsigned int modVal;
    int from = 0, to = 1;

    clSetKernelArg(multiKernel,0,sizeof(cl_mem),(void *)&menVector1);
    clSetKernelArg(multiKernel,1,sizeof(cl_mem),(void *)&menVector2);
    clSetKernelArg(multiKernel,2,sizeof(cl_mem),(void *)&(output[from]));
    
    clEnqueueNDRangeKernel(command, multiKernel,1,NULL,workItemCount,localItemCout,0,NULL,NULL);
    while (workItemCount[0] > 1){
        
        //Padding and re-caculate the glolab work size if needed
        modVal = workItemCount[0] % localItemCout[0];
        if (modVal != 0) workItemCount[0] += localItemCout[0] - modVal;

        clSetKernelArg(reduceKernel,0,sizeof(cl_mem),(void *)&(output[from]));
        clSetKernelArg(reduceKernel,1,sizeof(cl_mem),(void *)&(output[to]));

        // ping-pong!
        from = 1 - from;
        to = 1 - to;

        clEnqueueNDRangeKernel(command,reduceKernel,1,NULL,workItemCount,localItemCout,0,NULL,NULL);
        workItemCount[0] = ceil(workItemCount[0] / double(localItemCout[0]));
    }
    
    clEnqueueReadBuffer(command,output[from],CL_TRUE,0,1*sizeof(double),result,0,NULL,NULL);
    
    free(device);
    return result[0];
}

void
cleanup(int signo)
{
    // clean device
    clReleaseProgram(program);
    clReleaseContext(context);
    clReleaseKernel(multiKernel);
    clReleaseKernel(reduceKernel);
    clReleaseCommandQueue(command);
    clReleaseMemObject(menVector1);
    clReleaseMemObject(menVector2);
    clReleaseMemObject(output[0]);
    clReleaseMemObject(output[1]);
    exit(0);
}
   

static double*
readFile(char* filename){
    unsigned int length;
    double* vector;
    FILE *filePtr;
    
    filePtr = fopen(filename, "r");
    if (filePtr != NULL){
        fread(&length, sizeof(unsigned int), 1, filePtr);

        if (length % LWS != 0) 
            GWS = length + LWS - length % LWS;
        else
            GWS = length;
        vector = new double[GWS];
        fread(vector, sizeof(double), length , filePtr);
    }else{
        fprintf(stderr, "Can't open file %s \n", filename);
        exit(-1);
    }
    fclose(filePtr);
    return vector;
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
    printf("Result: %f\n", result);
    printf("%s\n", "-----------");
    cleanup(SIGUSR1);
    return 0;
}
                  
