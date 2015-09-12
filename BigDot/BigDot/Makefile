#simple makefile for Opencl on SoC computer


#scompiler
CC	= g++


#Include
CUDA = -I /usr/local/cuda-5.5/targets/x86_64-linux/include
NV = -L/usr/lib/nvidia-current
CL = -lOpenCl



LDFLAGS = $(CUDA) $(CUDA)
CFLAGS =  $(CL)

PROJECT = bigdot

OBJECTS = bigdot.o  

  
all: ${PROJECT}
${PROJECT}: ${OBJECTS}
	${CC} -o ${PROJECT} ${OBJECTS} ${LDFLAGS}

%.o: %.cpp
	${CC} -c ${LDFLAGS} *.cpp ${CFLAGS}

clean:
	rm -f core.* *.o *~ ${PROJECT}
