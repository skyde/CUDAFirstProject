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

	right[index].Self.Value = value + right[index].Bias.Value;
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

	double leftValue = left[index].Self.Value;

	for(int i = 0; i < rightLength; i++)
	{
		int w = index * rightLength + i;
		double rightDerivative = right[i].Self.Derivative;

		weights[w].Derivative = leftValue * rightDerivative;//weights[w].Value * right[i].Self.Derivative;

		total += weights[w].Value * rightDerivative;
	}

	left[index].Bias.Derivative = total;
	left[index].Self.Derivative = total;
}

__global__ void CaculateDerivativesFromDifferencePass(
		Node* values,
		double* targets)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	double v = targets[index] - values[index].Self.Value;

	values[index].Self.Derivative = v * 0.01;
}

__global__ void IterateWeightDerivativePass(Element* weights)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	weights[index].Value += weights[index].Derivative;
}

__global__ void IterateNodeDerivativePass(Node* nodes)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

//	nodes[index].Self.Value += nodes[index].Self.Derivative;
	nodes[index].Bias.Value += nodes[index].Bias.Derivative;
}

void Forward(NeuralNetwork* n)
{
	for(int i = 0; i < n->Layers.size() - 1; ++i)
	{
//		cout << i << " " << i + 1 << "\n";

		ForwardPass<<<N / M, M>>>(
				n->Layers[i]->DeviceData,
				n->Layers[i]->Length,
				n->Weights[i]->DeviceData,
				n->Layers[i + 1]->DeviceData,
				n->Layers[i + 1]->Length);
	}
}

void CaculateDerivativesFromDifference(NeuralNetwork* n, SharedData<double>* targetValues)
{
	SharedData<Node>* layer = n->Layers[n->Layers.size() - 1];
	int length = layer->Length;

	CaculateDerivativesFromDifferencePass<<<length, 1>>>(
			layer->DeviceData,
			targetValues->DeviceData);
}

void Backward(NeuralNetwork* n)
{
	for(int i = n->Layers.size() - 2; i >= 0; --i)
	{
		BackwardPass<<<N / M, M>>>(
				n->Layers[i]->DeviceData,
				n->Layers[i]->Length,
				n->Weights[i]->DeviceData,
				n->Layers[i + 1]->DeviceData,
				n->Layers[i + 1]->Length);
	}
}

void IterateDerivative(NeuralNetwork* n)
{
	for(int i = 0; i < n->Weights.size(); ++i)
	{
		int length = n->Weights[i]->Length;
		IterateWeightDerivativePass<<<length, 1>>>(
				n->Weights[i]->DeviceData);
	}

	// Don't iterate input or output layer
	for(int i = 1; i < n->Layers.size() - 1; ++i)
	{
		int length = n->Layers[i]->Length;
		IterateNodeDerivativePass<<<length, 1>>>(
				n->Layers[i]->DeviceData);
	}
}

int main(int argc, char **argv)
{
	NeuralNetwork* n = new NeuralNetwork();

	n->Layers[0]->HostData[0].Self.Value = 1;
	n->Layers[1]->HostData[0].Self.Value = 0;
	n->Layers[2]->HostData[0].Self.Value = 0;

//	n->Layers[1]->HostData[0].Self.Derivative = 0.01;
//	n->Layers[1]->HostData[1].Self.Derivative = 0.001;

	n->Weights[0]->HostData[0].Value = 1;
	n->Weights[1]->HostData[0].Value = 0.5;
//	n->Weights[0]->HostData[2].Value = -0.5;
//	n->Weights[0]->HostData[2].Value = 0;
//	n->Weights[0]->HostData[3].Value = 1;

	n->CopyToDevice();

//	double targetValues[2] = { 4.0, -2.0 };

	SharedData<double>* targetValues = new SharedData<double>(N);
	targetValues->HostData[0] = 12;
//	targetValues->HostData[1] = -2;
	targetValues->CopyToDevice();

//	cout << "Copy to device calls after initiated\n";

	for(int i = 0; i < 100; ++i)
	{
		cout << "\n";
		cout << "Epoch " << i;
		cout << "\n";

		Forward(n);
		CaculateDerivativesFromDifference(n, targetValues);
		Backward(n);
		IterateDerivative(n);

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
    delete targetValues;

    cout << "dispose finished\n";

    cudaDeviceReset();

    cout << "cudaDeviceReset finished\n";

    cout << "will exit\n";

//    exit(EXIT_SUCCESS);

	return 0;
}


