
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

__global__ void ForwardPass(
		double* source,
		double* weights,
		double* target,
		double* biases)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	double output = source[index];

	for(int i = 0; i < N; i++)
	{
		output *= weights[index * N + i];
	}

	target[index] = tanh(output + biases[index]);
}

//__global__ void BackwardPass(double* source, double* weights, double* target)
//{
//	int index = threadIdx.x + blockIdx.x * blockDim.x;
//
//	double output = source[index];
//
//	for(int i = 0; i < N; i++)
//	{
//		output *= weights[index * N + i];
//	}
//
//	target[index] = tanh(output);
//}

int main(int argc, char **argv)
{
	printf ("N = %d \n", N);

	SharedData* sourceLayer = new SharedData(N);
	SharedData* weights = new SharedData(N * N);
	SharedData* nextLayer = new SharedData(N);
	SharedData* nextLayerBiases = new SharedData(N);

	randomValues(sourceLayer->HostData, sourceLayer->Length);
	randomValues(weights->HostData, weights->Length);
	randomValues(nextLayerBiases->HostData, nextLayerBiases->Length);

	cout << "Generated random values\n";

	sourceLayer->CopyToDevice();
	weights->CopyToDevice();
	nextLayerBiases->CopyToDevice();

	cout << "Copy to device calls after initiated\n";

//	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
//	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

//	dim3 threadsPerBlock(16, 16);
//	dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

	ForwardPass<<<N / M, M>>>(
			sourceLayer->DeviceData,
			weights->DeviceData,
			nextLayer->DeviceData,
			nextLayerBiases->DeviceData);

	cout << "RunPass initiated\n";

	nextLayer->CopyToHost();

	cout << "CopyToHost initiated\n";

    cudaDeviceSynchronize();

	cout << "cudaDeviceSynchronize finished\n";

    getLastCudaError("Device kernel execution failed.\n");

    cout << "Execution finished, will print\n";

	for(int i = 0; i < nextLayer->Length && i < 512; i++)
	{
		cout << sourceLayer->HostData[i] << " = " << nextLayer->HostData[i] << "\n";
	}

    cout << "print finished\n";

	sourceLayer->Dispose();
	weights->Dispose();
	nextLayer->Dispose();
	nextLayerBiases->Dispose();

    cout << "dispose finished\n";

    cudaDeviceReset();

    cout << "cudaDeviceReset finished\n";

	delete sourceLayer;
	delete weights;
	delete nextLayer;
	delete nextLayerBiases;

    cout << "will exit\n";

//    exit(EXIT_SUCCESS);

	return 0;
}

void randomValues(double* a, int n)
{
	int i;
	for (i = 0; i < n; ++i)
	{
		a[i] = 0.5 + (double)rand() / RAND_MAX;

		cout << "random = " << a[i] << "\n";
	}
}
