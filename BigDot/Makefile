

CC = g++
OBJS = $(addsuffix .o, $(basename $(shell find *.cpp)))
DEPS = $(OBJS:.o=.d)

INCLUDE = /usr/local/cuda/include
LDFLAGS = $(libgl) $(libcl) -lm -lXmu
CXXFLAGS = -Wno-deprecated-declarations -Wall  -pedantic -ansi

PORJECT = bigdot 

ifeq ($(shell uname -s), Darwin)
	libgl = -framework OpenGL -framework GLUT
	libcl = -framework OpneCL
else
	libgl = -lGL -lGLU -lglut -lX11 -lGLEW 
	libcl = -lOpenCL
endif


$(PORJECT) : $(OBJS)
	$(CC) -I $(INCLUDE) $^ $(LDFLAGS) -o $@

$(OBJS): %.o: %.cpp
	$(CC) -I  $(CXXFLAGS) -c $^ -o $@ 

$(DEPS): %.d: %.cpp
	@echo "Generating "$@
	@set -e; rm -f $@; \
		g++ -MM $(CXXFLAGS) $< > $@.$$$$; \
		sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
		rm -f $@.$$$$

。PHONY: tar
tar:
	$(make clean)
	tar zcfv $(NAME).tar.gz *.* Makefile

。PHONY: clean
clean:
	rm -f $(OBJS) 
	rm -f $(DEPS)
	rm -f $(PORJECT)
