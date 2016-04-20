#include <iostream>
using namespace std;

template <class T> class SharedData
{
public:
	SharedData(int length) : Length(length), TotalBytes(length * sizeof(T))
	{
		HostData = (T *) malloc(Length);
		cudaMalloc((void **)& DeviceData, TotalBytes);

		cout << "construct" << "\n";
	}

	virtual ~SharedData()
	{
		free(HostData);
		cudaFree(DeviceData);

		cout << "deconstruct" << "\n";
	}

	void CopyToDevice()
	{
		cudaMemcpy(DeviceData, HostData, TotalBytes, cudaMemcpyHostToDevice);
	}

	void CopyToHost()
	{
		cudaMemcpy(HostData, DeviceData, TotalBytes, cudaMemcpyDeviceToHost);
	}

	const int TotalBytes;
	const int Length;

	T *HostData;
	T *DeviceData;
};
