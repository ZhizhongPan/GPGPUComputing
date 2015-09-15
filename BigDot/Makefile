#!/bin/sh

CC = g++
SRC = $(wildcard *.cpp)
OBJS = $(SRC:.c = .o)
DEPS = $(OBJS:.o=.d)

LDFLAGS = $(libgl) $(libcl) -lm -lXmu
CXXFLAGS = -Wno-deprecated-declarations -Wall -Wno-c++11-long-long -pedantic -ansi

PORJECT = bigdot 

ifeq ($(shell uname -s), Darwin)
	libgl = -framework OpenGL -framework GLUT
	libcl = -framework OpneCL
else
	libgl = -lGL -lGLU -lglut -lX11 -lGLEW 
	libcl = -lOpenCL
endif


$(PORJECT) : $(OBJS)
	$(CC) $(LDFLAGS) $^ -o $@

$(OBJS): %.o: %.cpp
	$(CC) $(CXXFLAGS) -c $< -o $@

$(DEPS): %.d: %.cpp
	@echo "Generating "$@
	@set -e; rm -f $@; \
		g++ -MM $(CXXFLAGS) $< > $@.$$$$; \
		sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
		rm -f $@.$$$$

。PHONY: tar
tar:
	$(clean)
	tar zcfv $(NAME).tar.gz *.* Makefile 

。PHONY: clean
clean:
	rm -f $(OBJS) 
	rm -f $(DEPS)
	rm -f $(PORJECT)