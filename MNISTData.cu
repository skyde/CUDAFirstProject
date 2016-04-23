#pragma once
#include <iostream>
using namespace std;
#include "helper_cuda.h"
#include "Globals.h"
#include "ReadMNIST.cu"
//#include "SharedData.cu"

class MNISTData
{
public:
	MNISTData(int index)
	{
		string fileName = "data" + to_string(index) + ".txt";

		cout << fileName << "\n";

		for(int i = 0; i < Elements.size(); ++i)
		{
			Elements[i] = new MNISTElement();
		}

		cout << "Start ReadMNISTData\n";
		ReadMNISTData(fileName, Elements);
		cout << "End ReadMNISTData\n";
	}

	virtual ~MNISTData()
	{

	}

//	array<SharedData<double>*>, MNIST_ELEMENTS_TO_LOAD> Values;

	array<MNISTElement*, MNIST_ELEMENTS_TO_LOAD> Elements;

//	SharedData<double>* Values = new SharedData<double>(MNIST_ELEMENT_SIZE);


//	void Dispose()
//	{
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
//
//	int TotalBytes;
//	int Length;
//
//	T *HostData;
//	T *DeviceData;
};
