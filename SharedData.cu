/*
// * Layer.h
// *
// *  Created on: Apr 20, 2016
// *      Author: admin
// */
//
//#ifndef LAYER_H_
//#define LAYER_H_

template <class T> class SharedData
{
public:
	SharedData(int length) : Length(length), TotalBytes(length * sizeof(T))
	{
		HostData = (T *) malloc(Length);
		cudaMalloc((void **)& DeviceData, TotalBytes);
	}

	virtual ~SharedData()
	{
		free(HostData);
		cudaFree(DeviceData);
	}

	const int TotalBytes;
	const int Length;

	T *HostData;
	T *DeviceData;
};

//#endif /* LAYER_H_ */
