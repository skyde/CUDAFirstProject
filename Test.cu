#include <stdio.h>
#include <iostream>
#include <array>
#include "helper_cuda.h"

#include "SharedData.cu"
#include "Layer.cu"
using namespace std;

//void initGL(int *argc, char **argv);

// Total Threads
#define N 2 // Nodes per layer
// Block Size
#define M 1 // 512

#define LAYERS 2

#define PRINT_DERIVATIVE false

//#define RADIUS 1


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
	Element() : Value(0), Derivative(0)
	{
//		cout << "Element ctor";
	}

    double Value, Derivative;

	public:
	void Print()
	{
		cout << "[" << Value;
#if PRINT_DERIVATIVE
		cout << " " << Derivative;
#endif
		cout << "]";
	}
};

struct __align__(sizeof(Element) * 2) Node
{
	Node() : Self(), Bias()
	{
//		cout << "Node ctor";
	}

	Element Self, Bias;

	public:
	void Print()
	{
		cout << "(" << Self.Value;
#if PRINT_DERIVATIVE
		cout << " " << Self.Derivative;
#endif
		cout << ")";

//		cout << "(" << Self.Value << " " << Self.Derivative << ")";
	}
};

void randomValues(double* a, int n);
void randomValues(Node* a, int n);
void randomValues(Element* a, int n);

__global__ void ForwardPass(
		Node* left,
		int leftLength,
		Element* weights, // left to right
		Node* right,
		int rightLength)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	double value = 0;//left[index].Self.Value;

	for(int i = 0; i < leftLength; i++)
	{
		int lookup = index + i * rightLength;

		value += left[i].Self.Value * weights[lookup].Value;
	}

	right[index].Self.Value = value;// + right[index].Bias.Value
}

__global__ void BackwardPass(
		Node* left,
		int leftLength,
		Element* weights,
		Node* right,
		int rightLength)
{
//	int index = threadIdx.x + blockIdx.x * blockDim.x;
//
//	double output = left[index].Self.Value;
//
//	for(int i = 0; i < N; i++)
//	{
//		output *= weights[index * N + i].Value;
//	}
//
//	right[index].Self.Value = tanh(output + right[index].Bias.Value);
}

int main(int argc, char **argv)
{
	printf ("N = %d \n", N);

	array<SharedData<Node>*, LAYERS> layers;
	array<SharedData<Element>*, LAYERS - 1> weights;

	for(int i = 0; i < layers.size(); i++)
	{
		layers[i] = new SharedData<Node>(N);

		if(i != 0)
		{
			int length = layers[i - 1]->Length * layers[i]->Length;

			weights[i - 1] = new SharedData<Element>(length);

			cout << "weights " << i - 1 << ", l = " << length << "\n";
		}
	}

	layers[0]->HostData[0].Self.Value = 1;
	layers[0]->HostData[1].Self.Value = 10;

//	layers[1]->HostData[0].Self.Value = 10;
//	layers[1]->HostData[1].Self.Value = 20;

	weights[0]->HostData[0].Value = 1;
	weights[0]->HostData[1].Value = 0.5;
	weights[0]->HostData[2].Value = 0;
	weights[0]->HostData[3].Value = 1;

	cout << "Generated random values\n";

	layers[0]->CopyToDevice();
	weights[0]->CopyToDevice();
	layers[1]->CopyToDevice();

	cout << "Copy to device calls after initiated\n";

	ForwardPass<<<N / M, M>>>(
			layers[0]->DeviceData,
			layers[0]->Length,
			weights[0]->DeviceData,
			layers[1]->DeviceData,
			layers[1]->Length);

    cudaDeviceSynchronize();

	BackwardPass<<<N / M, M>>>(
			layers[0]->DeviceData,
			layers[0]->Length,
			weights[0]->DeviceData,
			layers[1]->DeviceData,
			layers[1]->Length);

	cout << "RunPass initiated\n";

	layers[1]->CopyToHost();

	cout << "CopyToHost initiated\n";

    cudaDeviceSynchronize();

	cout << "cudaDeviceSynchronize finished\n";

    getLastCudaError("Device kernel execution failed.\n");

    cout << "Execution finished, will print\n";

//    int numLayers = 2;
//    int nodesPerLayer = N;

	for(int y = 0; y < N; y++)
	{
		for(int x = 0; x < LAYERS; x++)
		{
			layers[x]->HostData[y].Print();
			cout << " ";

			int nextLayer = x + 1;
			if(nextLayer < LAYERS)
			{
				int block = layers[nextLayer]->Length;
				for(int w = 0; w < block; w++)
				{
					int index = y * block + w;

					cout << y << "->" << w;

					weights[x]->HostData[index].Print();

					cout << " ";
				}
			}
		}

		cout << "\n";
	}

//	for(int i = 0; i < layer0->Length && i < 512; i++)
//	{
//		cout << "Node.Self " << layer0->HostData[i].Self.Value << " Derivative " << layer0->HostData[i].Self.Derivative << "\n";
//		cout << "Node.Bias " << layer0->HostData[i].Bias.Value << " Derivative " << layer0->HostData[i].Bias.Derivative << "\n";
//
//		for(int x = 0; x < layer1->Length; x++)
//		{
//			int w = i * layer1->Length + x;
//
//			cout << "Weight.Self " << weights->HostData[w].Value << " Derivative " << weights->HostData[w].Derivative << "\n";
//		}
//	}

    cout << "print finished\n";

//	leftValues->Dispose();
//	weights->Dispose();
//	rightValues->Dispose();
//	rightBiases->Dispose();

	for(int i = 0; i < layers.size(); i++)
	{
		delete layers[i];
	}

	for(int i = 0; i < weights.size(); i++)
	{
		delete weights[i];
	}

//	delete layers[1];
//	delete weight0to1;
//	delete layer1;

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

