#include <stdio.h>
#include <iostream>
#include "helper_cuda.h"

#include "SharedData.cu"
#include "Layer.cu"
using namespace std;

//void initGL(int *argc, char **argv);

// Total Threads
#define N 2 // 4096
// Block Size
#define M 2 // 512

#define RADIUS 1


//__global__ void add(int *a, int *b, int *c, int n)
//{
//	int index = threadIdx.x + blockIdx.x * blockDim.x;
//
//	if(index < n)
//	{
//		c[index] = a[index] + b[index];
//	}
//}

struct __align__(sizeof(double) * 2) Element
{
    double Value, Derivative;
};

struct __align__(sizeof(Element) * 2) Node
{
	Element Self, Bias;
};

void randomValues(double* a, int n);
void randomValues(Node* a, int n);
void randomValues(Element* a, int n);

__global__ void ForwardPass(
		Node* left,
		Element* weights, // left to right
		Node* right)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	double output = left[index].Self.Value;

	for(int i = 0; i < N; i++)
	{
		output *= weights[index * N + i].Value;
	}

	right[index].Self.Value = tanh(output + right[index].Bias.Value);
}

int main(int argc, char **argv)
{
	printf ("N = %d \n", N);

	SharedData<Node>* layer0 = new SharedData<Node>(N);
	SharedData<Element>* weights = new SharedData<Element>(N * N);
	SharedData<Node>* layer1 = new SharedData<Node>(N);

	randomValues(layer0->HostData, layer0->Length);
	randomValues(weights->HostData, weights->Length);
	randomValues(layer1->HostData, layer1->Length);

	cout << "Generated random values\n";

	layer0->CopyToDevice();
	weights->CopyToDevice();
	layer1->CopyToDevice();

	cout << "Copy to device calls after initiated\n";

	ForwardPass<<<N / M, M>>>(
			layer0->DeviceData,
			weights->DeviceData,
			layer1->DeviceData);

	cout << "RunPass initiated\n";

	layer1->CopyToHost();

	cout << "CopyToHost initiated\n";

    cudaDeviceSynchronize();

	cout << "cudaDeviceSynchronize finished\n";

    getLastCudaError("Device kernel execution failed.\n");

    cout << "Execution finished, will print\n";

	for(int i = 0; i < layer0->Length && i < 512; i++)
	{
		cout << "Node.Self " << layer0->HostData[i].Self.Value << " Derivative " << layer0->HostData[i].Self.Derivative << "\n";
		cout << "Node.Bias " << layer0->HostData[i].Bias.Value << " Derivative " << layer0->HostData[i].Bias.Derivative << "\n";

		for(int x = 0; x < layer1->Length; x++)
		{
			int w = i * layer1->Length + x;

			cout << "Weight.Self " << weights->HostData[w].Value << " Derivative " << weights->HostData[w].Derivative << "\n";
		}
	}

    cout << "print finished\n";

//	leftValues->Dispose();
//	weights->Dispose();
//	rightValues->Dispose();
//	rightBiases->Dispose();

	delete layer0;
	delete weights;
	delete layer1;

    cout << "dispose finished\n";

    cudaDeviceReset();

    cout << "cudaDeviceReset finished\n";

    cout << "will exit\n";

//    exit(EXIT_SUCCESS);

	return 0;
}

inline double randomValue()
{
	return 0.5 + (double)rand() / RAND_MAX;
}

void randomValues(double* a, int n)
{
	int i;
	for (i = 0; i < n; ++i)
	{
		a[i] = randomValue();

//		cout << "random = " << a[i] << "\n";
	}
}

void randomValues(Node* a, int n)
{
	int i;
	for (i = 0; i < n; ++i)
	{
		a[i].Self.Value = randomValue();
		a[i].Bias.Value = randomValue();
	}
}

void randomValues(Element* a, int n)
{
	int i;
	for (i = 0; i < n; ++i)
	{
		a[i].Value = randomValue();
	}
}

