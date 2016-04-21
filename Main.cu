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

void Forward(NeuralNetwork* n)
{
	ForwardPass<<<N / M, M>>>(
			n->Layers[0]->DeviceData,
			n->Layers[0]->Length,
			n->Weights[0]->DeviceData,
			n->Layers[1]->DeviceData,
			n->Layers[1]->Length);
}
void Backward(NeuralNetwork* n)
{
	BackwardPass<<<N / M, M>>>(
			n->Layers[0]->DeviceData,
			n->Layers[0]->Length,
			n->Weights[0]->DeviceData,
			n->Layers[1]->DeviceData,
			n->Layers[1]->Length);
}

int main(int argc, char **argv)
{
	NeuralNetwork* n = new NeuralNetwork();

	n->Layers[0]->HostData[0].Self.Value = 1;
	n->Layers[0]->HostData[1].Self.Value = 2;

	n->Layers[1]->HostData[0].Self.Derivative = 0.01;
	n->Layers[1]->HostData[1].Self.Derivative = 0.001;

	n->Weights[0]->HostData[0].Value = 1;
	n->Weights[0]->HostData[1].Value = 0.5;
	n->Weights[0]->HostData[2].Value = 0;
	n->Weights[0]->HostData[3].Value = 1;

	n->CopyToDevice();

//	cout << "Copy to device calls after initiated\n";

	for(int i = 0; i < 10; ++i)
	{
		cout << "\n";
		cout << "Epoch " << i;
		cout << "\n";

		Forward(n);
		Backward(n);

		n->CopyToHost();

	    cudaDeviceSynchronize();

	    n->Print();
	}

//	cout << "RunPass initiated\n";


//	cout << "CopyToHost initiated\n";

//    cudaDeviceSynchronize();

//	cout << "cudaDeviceSynchronize finished\n";

//    getLastCudaError("Device kernel execution failed.\n");

//    cout << "Execution finished, will print\n";

//    cout << "print finished\n";

    delete n;

    cout << "dispose finished\n";

    cudaDeviceReset();

    cout << "cudaDeviceReset finished\n";

    cout << "will exit\n";

//    exit(EXIT_SUCCESS);

	return 0;
}


