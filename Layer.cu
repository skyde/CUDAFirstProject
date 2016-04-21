#pragma once

#include <iostream>
using namespace std;
#include "helper_cuda.h"
#include "SharedData.cu"

//template <class T>
class Layer
{
public:
	Layer(int length)
	{
		Values = new SharedData<double>(length);
		Biases = new SharedData<double>(length);
		Derivatives = new SharedData<double>(length);
	}

	virtual ~Layer()
	{
	}

	SharedData<double>* Values;
	SharedData<double>* Biases;
	SharedData<double>* Derivatives;

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
