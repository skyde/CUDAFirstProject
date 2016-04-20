#include <iostream>
using namespace std;
#include "helper_cuda.h"

//template <class T>
class SharedData
{
public:
	SharedData(int length)
	{
		Length = length;
		TotalBytes = length * sizeof(double);

		HostData = (double *) malloc(TotalBytes);
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

	int TotalBytes;
	int Length;

	double *HostData;
	double *DeviceData;
};
