#include <iostream>
using namespace std;
#include "helper_cuda.h"

template <class T> class SharedData
{
public:
	SharedData(int length) : Length(length), TotalBytes(length * sizeof(T))
	{
		HostData = (T *) malloc(Length);
		checkCudaErrors(cudaMalloc((void **)& DeviceData, TotalBytes));

		cout << "construct" << "\n";
	}

	virtual ~SharedData()
	{
	}

	void Dispose()
	{
		free(HostData);
		cudaFree(DeviceData);

		cout << "deconstruct" << "\n";
	}

	void CopyToDevice()
	{
		checkCudaErrors(cudaMemcpy(DeviceData, HostData, TotalBytes, cudaMemcpyHostToDevice));
	}

	void CopyToHost()
	{
		checkCudaErrors(cudaMemcpy(HostData, DeviceData, TotalBytes, cudaMemcpyDeviceToHost));
	}

	const int TotalBytes;
	const int Length;

	T *HostData;
	T *DeviceData;
};
