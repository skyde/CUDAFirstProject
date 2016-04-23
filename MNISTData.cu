#pragma once

#include <iostream>
using namespace std;
#include "helper_cuda.h"
#include "Globals.h"
#include "ReadMNIST.cpp"
#include "SharedData.cu"

class MNISTElement
{
	public:
	MNISTElement(double* values)
	{
		Values = new SharedData<double>(MNIST_ELEMENT_SIZE);

//		cout << "Create element\n";

		for(int i = 0; i < MNIST_ELEMENT_SIZE; ++i)
		{
			Values->HostData[i] = values[i];
		}

//		Values->HostData[2] = 5;
		Values->CopyToDevice();
	}

	SharedData<double>* Values;

	virtual ~MNISTElement()
	{
//		Values->
	}
};

class MNISTData
{
public:
	MNISTData(int index)
	{
		string fileName = "data" + to_string(index) + ".txt";

		cout << fileName << "\n";

		array<double*, MNIST_ELEMENTS_TO_LOAD> elements = ReadMNISTData(fileName);

		for(int i = 0; i < elements.size(); ++i)
		{
//			MNISTElement e = ;

			Elements[i] = new MNISTElement(elements[i]);
		}
	}

	virtual ~MNISTData()
	{

	}

//	array<SharedData<double>*>, MNIST_ELEMENTS_TO_LOAD> Values;

	array<MNISTElement*, MNIST_ELEMENT_SIZE> Elements;

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
