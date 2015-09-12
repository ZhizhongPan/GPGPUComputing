//
//  testVectorGen.cpp
//  BigDot
//
//  Created by zhizhop on 15/9/11.
//  Copyright (c) 2015年 Clemson. All rights reserved.
//

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>

#define LENGTH 257

static double
dotProd(double* v1, double* v2){
    unsigned int i;
    double result = 0;
    for (i = 0; i < LENGTH; i++) {
        result += v1[i] * v2[i];
    }
    return result;
}

static int
writeFile(char* filename, double* vector){
    FILE *filePtr;
    
    unsigned int length = LENGTH;
    filePtr = fopen(filename, "w+");
    if (filePtr != NULL) {
        fwrite(&length, sizeof(unsigned int), 1, filePtr);
        fwrite(vector, sizeof(double), LENGTH, filePtr);
    }else{
        fprintf(stderr, "Can't open file %s \n", filename);
        exit(-1);
    }
    fclose(filePtr);
    return 0;
}



int main(int argc, char* argv[]){
    
    double vector1[LENGTH] = {1.0};
    double vector2[LENGTH] = {1.0};
    
    writeFile(argv[1], vector1);
    writeFile(argv[2], vector2);
    
    double result = dotProd(vector1, vector2);
    printf("The Result is %f:", result);
    
    
}