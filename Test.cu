
//#include "GL/glut.h"
// OpenGL Graphics includes
//#include <GL/glew.h>
//#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
//#include <GL/wglew.h>
//#endif
//#if defined(__APPLE__) || defined(__MACOSX)
//  #pragma clang diagnostic ignored "-Wdeprecated-declarations"
//  #include <GLUT/glut.h>
//  #ifndef glutCloseFunc
//  #define glutCloseFunc glutWMCloseFunc
//  #endif
//#else
//#include <GL/freeglut.h>
//#endif
//
//// CUDA runtime
//// CUDA utilities and system includes
//#include <cuda_runtime.h>
//#include <cuda_gl_interop.h>
//
//#include <helper_functions.h>
//#include <helper_cuda.h>
//#include <helper_cuda_gl.h>
//#include <rendercheck_gl.h>


#include <stdio.h>
#include <iostream>

#include "SharedData.cu"
using namespace std;

void random_ints(int* a, int n);
//void initGL(int *argc, char **argv);

// Total Threads
#define N (4096 * 8)
// Block Size
#define M 512

#define RADIUS 1

class Layer
{
public:

};

//__global__ void add(int *a, int *b, int *c, int n)
//{
//	int index = threadIdx.x + blockIdx.x * blockDim.x;
//
//	if(index < n)
//	{
//		c[index] = a[index] + b[index];
//	}
//}

__global__ void stencil_1d(int* in, int* out)
{
	__shared__ int temp[M + 2 * RADIUS];
	int gindex = threadIdx.x + blockIdx.x * blockDim.x;
	int lindex = threadIdx.x + RADIUS;

	temp[lindex] = in[gindex];

	if(threadIdx.x < RADIUS)
	{
		temp[lindex - RADIUS] = in[gindex - RADIUS];
		temp[lindex + M] = in[gindex + M];
	}

	__syncthreads();

	int result = 0;

	for	(int offset = -RADIUS; offset <= RADIUS; offset++)
	{
		result += temp[lindex + offset];
	}

	out[gindex] = result;
}

//template <class T>
//class Stack {
//};

int main(int argc, char **argv)
{
//	initGL(&argc, argv);
	printf ("N = %d \n", N);

	SharedData<int>* a = new SharedData<int>(N);
	SharedData<int>* b = new SharedData<int>(N);
//	SharedData<int>* c = new SharedData<int>(N);

//	double *inputs;
//	double *outputs;

//	int *a, *b, *c;
//	int *d_a, *d_b, *d_c;
//	int size = N * sizeof(int);

//	cudaMalloc((void **)&d_a, size);
//	cudaMalloc((void **)&d_b, size);
//	cudaMalloc((void **)&d_c, size);
//
//	a = (int *)malloc(size);
//	b = (int *)malloc(size);
//	c = (int *)malloc(size);

	random_ints(a->HostData, a->Length);
//	random_ints(b->HostData, N);

	a->CopyToDevice();
//	b->CopyToDevice();

//	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
//	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	stencil_1d<<<((N + M - 1) / M), M>>>(a->DeviceData, b->DeviceData);

	b->CopyToHost();
//	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

	for(int i = 0; i < b->Length && i < 512; i++)
	{
		printf ("%d = %d \n", a->HostData[i], b->HostData[i]);
	}

//	free(a);
//	free(b);
//	free(c);

//	cudaFree(d_a);
//	cudaFree(d_b);
//	cudaFree(d_c);

	delete a;
	delete b;

    cudaDeviceReset();
    exit(EXIT_SUCCESS);

	return 0;
}
//
//void initGL(int *argc, char **argv)
//{
////    printf("Initializing GLUT...\n");
////    glutInit(argc, argv);
////
////    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
////    glutInitWindowSize(1024, 768);
////    glutInitWindowPosition(0, 0);
////    glutCreateWindow(argv[0]);
////
//////    glutDisplayFunc(displayFunc);
//////    glutKeyboardFunc(keyboardFunc);
//////    glutMouseFunc(clickFunc);
//////    glutMotionFunc(motionFunc);
//////    glutReshapeFunc(reshapeFunc);
//////    glutTimerFunc(REFRESH_DELAY, timerEvent, 0);
//////    initMenus();
////
////    printf("Loading extensions: %s\n", glewGetErrorString(glewInit()));
////
////    if (!glewIsSupported("GL_VERSION_1_5 GL_ARB_vertex_buffer_object GL_ARB_pixel_buffer_object"))
////    {
////        fprintf(stderr, "Error: failed to get minimal extensions for demo\n");
////        fprintf(stderr, "This sample requires:\n");
////        fprintf(stderr, "  OpenGL version 1.5\n");
////        fprintf(stderr, "  GL_ARB_vertex_buffer_object\n");
////        fprintf(stderr, "  GL_ARB_pixel_buffer_object\n");
////        exit(EXIT_SUCCESS);
////    }
////
////    printf("OpenGL window created.\n");
//}
void random_ints(int* a, int n)
{
	int i;
	for (i = 0; i < n; ++i)
	{
		a[i] = rand() % 4;
	}
}
