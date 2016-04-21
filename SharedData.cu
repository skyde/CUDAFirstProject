#pragma once

#include <iostream>
using namespace std;
#include "helper_cuda.h"

template <class T>
class SharedData
{
public:
	SharedData(int length)
	{
		Length = length;
		TotalBytes = length * sizeof(T);

		HostData = (T *) calloc(1, TotalBytes);
		checkCudaErrors(cudaMalloc((void **)& DeviceData, TotalBytes));

//		for(int i = 0; i < length; i++)
//		{
//			HostData[i] = new T();
//		}

//		cout << "construct" << "\n";
	}

	virtual ~SharedData()
	{
		free(HostData);
		cudaFree(DeviceData);

//		cout << "deconstruct" << "\n";
	}

//	void Dispose()
//	{
//	}

	void CopyToDevice()
	{
		checkCudaErrors(cudaMemcpy(DeviceData, HostData, TotalBytes, cudaMemcpyHostToDevice));
	}

	void CopyToHost()
	{
		checkCudaErrors(cudaMemcpy(HostData, DeviceData, TotalBytes, cudaMemcpyDeviceToHost));
	}

	int TotalBytes;
	int Length;

	T *HostData;
	T *DeviceData;
};
