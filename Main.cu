#include <stdio.h>
#include <iostream>
#include <array>
using namespace std;

#include "NeuralNetwork.cu"
__global__ void ForwardPass(
		Node* left,
		int leftLength,
		Element* weights,
		Node* right,
		int rightLength)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	double value = 0;

	for(int i = 0; i < leftLength; i++)
	{
		int w = index + i * rightLength;

		value += left[i].Self.Value * weights[w].Value;
	}

	right[index].Self.Value = value;
}

__global__ void BackwardPass(
		Node* left,
		int leftLength,
		Element* weights,
		Node* right,
		int rightLength)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	double total = 0;

	for(int i = 0; i < rightLength; i++)
	{
		int w = index * rightLength + i;

		weights[w].Derivative = weights[w].Value * right[i].Self.Derivative;

		total += weights[w].Derivative;
	}

	left[index].Self.Derivative = total;
}

int main(int argc, char **argv)
{
//	printf ("N = %d \n", N);

	NeuralNetwork* n = new NeuralNetwork();

//	network->

	n->layers[0]->HostData[0].Self.Value = 1;
	n->layers[0]->HostData[1].Self.Value = 2;

	n->layers[1]->HostData[0].Self.Derivative = 1;
	n->layers[1]->HostData[1].Self.Derivative = 0.1;

	n->weights[0]->HostData[0].Value = 1;
	n->weights[0]->HostData[1].Value = 0.5;
	n->weights[0]->HostData[2].Value = 0;
	n->weights[0]->HostData[3].Value = 1;

//	cout << "Generated random values\n";
	n->CopyToDevice();

//	layers[0]->CopyToDevice();
//	weights[0]->CopyToDevice();
//	layers[1]->CopyToDevice();

	cout << "Copy to device calls after initiated\n";


	ForwardPass<<<N / M, M>>>(
			n->layers[0]->DeviceData,
			n->layers[0]->Length,
			n->weights[0]->DeviceData,
			n->layers[1]->DeviceData,
			n->layers[1]->Length);

	BackwardPass<<<N / M, M>>>(
			n->layers[0]->DeviceData,
			n->layers[0]->Length,
			n->weights[0]->DeviceData,
			n->layers[1]->DeviceData,
			n->layers[1]->Length);
//	n->Forward();
//	n->Backward();

	cout << "RunPass initiated\n";
//    cudaDeviceSynchronize();
	n->CopyToHost();
//
//	layers[0]->CopyToHost();
//	weights[0]->CopyToHost();
//	layers[1]->CopyToHost();

	cout << "CopyToHost initiated\n";

    cudaDeviceSynchronize();

	cout << "cudaDeviceSynchronize finished\n";

    getLastCudaError("Device kernel execution failed.\n");

    cout << "Execution finished, will print\n";

    n->Print();
    cout << "print finished\n";

    delete n;

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


