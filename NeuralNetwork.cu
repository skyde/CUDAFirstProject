#pragma once
#include <iostream>
using namespace std;
#include "helper_cuda.h"
#include "SharedData.cu"

template <class T>
class NeuralNetwork
{
public:
	NeuralNetwork()
	{

	}

	virtual ~NeuralNetwork()
	{
	}

//	SharedData<T>* Values;
//	SharedData<T>* Biases;
//	SharedData<T>* Derivatives;

//	void Dispose()
//	{
//		free(HostData);
//		cudaFree(DeviceData);
//
//		cout << "deconstruct" << "\n";
//	}

//	void CopyToDevice()
//	{
//		checkCudaErrors(cudaMemcpy(DeviceData, HostData, TotalBytes, cudaMemcpyHostToDevice));
//	}
//
//	void CopyToHost()
//	{
//		checkCudaErrors(cudaMemcpy(HostData, DeviceData, TotalBytes, cudaMemcpyDeviceToHost));
//	}

//	int TotalBytes;
//	int Length;
//
//	T *HostData;
//	T *DeviceData;
};
