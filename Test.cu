
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
#include "helper_cuda.h"

#include "SharedData.cu"
using namespace std;

void randomValues(double* a, int n);
//void initGL(int *argc, char **argv);

// Total Threads
#define N 16 // 4096
// Block Size
#define M 2 // 512

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

__global__ void RunPass(double* source, double* weights, double* target)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	double output = source[index];

	for(int i = 0; i < N; i++)
	{
		output *= weights[index * N + i];
	}

	target[index] = output * 2;
//	__shared__ int temp[M + 2 * RADIUS];
//	int gindex = threadIdx.x + blockIdx.x * blockDim.x;
//	int lindex = threadIdx.x + RADIUS;
//
//	temp[lindex] = in[gindex];
//
//	if(threadIdx.x < RADIUS)
//	{
//		temp[lindex - RADIUS] = in[gindex - RADIUS];
//		temp[lindex + M] = in[gindex + M];
//	}
//
//	__syncthreads();
//
//	int result = 0;
//
//	for	(int offset = -RADIUS; offset <= RADIUS; offset++)
//	{
//		result += temp[lindex + offset];
//	}
//
//	out[gindex] = result;
}

//template <class T>
//class Stack {
//};

int main(int argc, char **argv)
{
//	initGL(&argc, argv);
	printf ("N = %d \n", N);

	SharedData* layer0 = new SharedData(N);
	SharedData* weights = new SharedData(N * N);
	SharedData* layer1 = new SharedData(N);

//    cudaDeviceSynchronize();

	randomValues(layer0->HostData, layer0->Length);
	randomValues(weights->HostData, weights->Length);
//	random_ints(b->HostData, N);

	cout << "Generated random values\n";

	layer0->CopyToDevice();
	weights->CopyToDevice();
//	b->CopyToDevice();
	cout << "Copy to device calls after initiated\n";

//	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
//	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

//	dim3 threadsPerBlock(16, 16);
//	dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

	RunPass<<<N / M, M>>>(layer0->DeviceData, weights->DeviceData, layer1->DeviceData);

	cout << "RunPass initiated\n";
//    cudaError_t err = cudaGetLastError();
//
//    if (cudaSuccess != err)
//    {
//        fprintf(stderr, "%s(%i) : getLastCudaError() CUDA error : %s : (%d) %s.\n",
//                file, line, "Execution on device failed", (int)err, cudaGetErrorString(err));
//        DEVICE_RESET
//        exit(EXIT_FAILURE);
//    }

//    cudaDeviceSynchronize();

	layer1->CopyToHost();

	cout << "CopyToHost initiated\n";
//	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
	cout << "cudaDeviceSynchronize finished\n";

    getLastCudaError("Device kernel execution failed.\n");

    cout << "Execution finished, will print\n";

	for(int i = 0; i < layer1->Length && i < 512; i++)
	{
		cout << layer0->HostData[i] << " = " << layer1->HostData[i] << "\n";
	}

    cout << "print finished\n";

	layer0->Dispose();
	weights->Dispose();
	layer1->Dispose();

    cout << "dispose finished\n";

//    cudaDeviceReset();
    cout << "cudaDeviceReset finished\n";
    exit(EXIT_SUCCESS);

	delete layer0;
	delete weights;
	delete layer1;

    cout << "will exit\n";

	return 0;
}

void randomValues(double* a, int n)
{
	int i;
	for (i = 0; i < n; ++i)
	{
		a[i] = 1.0 + (double)rand() / RAND_MAX;

		cout << "random = " << a[i] << "\n";
	}
}
