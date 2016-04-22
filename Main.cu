#include <stdio.h>
#include <iostream>
#include <array>
#include "NeuralNetwork.cu"
using namespace std;

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

	value += right[index].Bias.Value;

	if(right[index].Activation == ActivationTanH)
	{
		value = tanh(value);
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

	double value = 0;

	double leftValue = left[index].Self.Value;

	for(int i = 0; i < rightLength; i++)
	{
		int w = index * rightLength + i;
		double rightDerivative = right[i].Self.Derivative;

		weights[w].Derivative = leftValue * rightDerivative;//weights[w].Value * right[i].Self.Derivative;

		value += weights[w].Value * rightDerivative;
	}

	left[index].Bias.Derivative = value;

	if(left[index].Activation == ActivationTanH)
	{
		value = 2.0 / (1.0 + pow(2.718281828459, -2.0 * value)) - 1.0;//tanh(value);
	}

	left[index].Self.Derivative = value;
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
	n->Layers[0]->HostData[1].Self.Value = 5;
	n->Layers[0]->HostData[2].Self.Value = -5;

	// Middle layers are activated
	for(int i = 0; i < n->Layers[1]->Length; ++i)
	{
		n->Layers[1]->HostData[i].Activation = ActivationTanH;
	}

	n->CopyToDevice();

//	double targetValues[2] = { 4.0, -2.0 };

//	array<int>* test;
//
//	cout << test.size();

	SharedData<double>* targetValues = new SharedData<double>(N);
	targetValues->HostData[0] = 12;
	targetValues->HostData[1] = -10;
	targetValues->HostData[2] = 5;
	targetValues->CopyToDevice();

//	cout << "Copy to device calls after initiated\n";

	cout << "\n";

	for(int i = 0; i < 500; ++i)
	{
		cout << "Epoch " << (i + 1);

		Forward(n);
		CaculateDerivativesFromDifference(n, targetValues);
		Backward(n);
		IterateDerivative(n);

		n->CopyToHost();

	    cudaDeviceSynchronize();

	    cout << ", error " << n->CaculateError(targetValues->HostData);

		cout << "\n";

		if(i == 0)
		{
		    n->PrintVerbose();
		}
	}

    n->PrintVerbose();

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


