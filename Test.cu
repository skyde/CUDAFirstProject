
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
		double* leftValues,
		double* weights, // left to right
		double* rightValues, // write target
		double* rightBiases)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	double output = leftValues[index];

	for(int i = 0; i < N; i++)
	{
		output *= weights[index * N + i];
	}

	rightValues[index] = tanh(output + rightBiases[index]);
}

__global__ void BackwardPass(
		double* leftValues,
		double* leftBiases,
		double* leftDerivatives,
		double* weights,
		double* rightValues,
		double* rightBiases,
		double* rightDerivatives)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	double output = leftValues[index];

	for(int i = 0; i < N; i++)
	{
		output *= weights[index * N + i];
	}

	rightValues[index] = tanh(output + rightBiases[index]);
}
//__global__ void BackwardPass(
//		double* left,
//		double* weights, // left to right
//		double* right,
//		double* biases)
//{
//	int index = threadIdx.x + blockIdx.x * blockDim.x;
//
//	double output = left[index];
//
//	for(int i = 0; i < N; i++)
//	{
//		output *= weights[index * N + i];
//	}
//
//	right[index] = tanh(output + biases[index]);
//}

int main(int argc, char **argv)
{
	printf ("N = %d \n", N);

	SharedData<double>* leftValues = new SharedData<double>(N);
	SharedData<double>* weights = new SharedData<double>(N * N);
	SharedData<double>* rightValues = new SharedData<double>(N);
	SharedData<double>* rightBiases = new SharedData<double>(N);

	randomValues(leftValues->HostData, leftValues->Length);
	randomValues(weights->HostData, weights->Length);
	randomValues(rightBiases->HostData, rightBiases->Length);

	cout << "Generated random values\n";

	leftValues->CopyToDevice();
	weights->CopyToDevice();
	rightBiases->CopyToDevice();

	cout << "Copy to device calls after initiated\n";

//	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
//	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

//	dim3 threadsPerBlock(16, 16);
//	dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

	ForwardPass<<<N / M, M>>>(
			leftValues->DeviceData,
			weights->DeviceData,
			rightValues->DeviceData,
			rightBiases->DeviceData);

	cout << "RunPass initiated\n";

	rightValues->CopyToHost();

	cout << "CopyToHost initiated\n";

    cudaDeviceSynchronize();

	cout << "cudaDeviceSynchronize finished\n";

    getLastCudaError("Device kernel execution failed.\n");

    cout << "Execution finished, will print\n";

	for(int i = 0; i < rightValues->Length && i < 512; i++)
	{
		cout << leftValues->HostData[i] << " = " << rightValues->HostData[i] << "\n";
	}

    cout << "print finished\n";

	leftValues->Dispose();
	weights->Dispose();
	rightValues->Dispose();
	rightBiases->Dispose();

    cout << "dispose finished\n";

    cudaDeviceReset();

    cout << "cudaDeviceReset finished\n";

	delete leftValues;
	delete weights;
	delete rightValues;
	delete rightBiases;

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
