#pragma once
#include <stdio.h>
#include <iostream>
#include <array>
#include "helper_cuda.h"
#include <stdlib.h>

#include "SharedData.cu"
#include "Layer.cu"
#include "Element.cu"
#include "Node.cu"
//#include "NeuralNetworkDevice.cu"
using namespace std;

// Total Threads
#define N 2 // Nodes per layer
// Block Size
#define M 1 // 512

#define LAYERS 2

#define PRINT_DERIVATIVE true

//template <class T>
class NeuralNetwork
{
public:
	NeuralNetwork()
	{
//		array<SharedData<Node>*, LAYERS> layers;
//		array<SharedData<Element>*, LAYERS - 1> weights;

		for(int i = 0; i < Layers.size(); i++)
		{
			Layers[i] = new SharedData<Node>(N);

			if(i != 0)
			{
				int length = Layers[i - 1]->Length * Layers[i]->Length;

				Weights[i - 1] = new SharedData<Element>(length);

				cout << "weights " << i - 1 << ", l = " << length << "\n";
			}
		}
	}

	array<SharedData<Node>*, LAYERS> Layers;
	array<SharedData<Element>*, LAYERS - 1> Weights;

//	void randomValues(double* a, int n);
//	void randomValues(Node* a, int n);
//	void randomValues(Element* a, int n);

//	void Forward()
//	{
//		ForwardPass<<<N / M, M>>>(
//				layers[0]->DeviceData,
//				layers[0]->Length,
//				weights[0]->DeviceData,
//				layers[1]->DeviceData,
//				layers[1]->Length);
//	}
//
//	void Backward()
//	{
//		BackwardPass<<<N / M, M>>>(
//				layers[0]->DeviceData,
//				layers[0]->Length,
//				weights[0]->DeviceData,
//				layers[1]->DeviceData,
//				layers[1]->Length);
//	}

	void CopyToDevice()
	{
		for(int i = 0; i < Layers.size(); ++i)
		{
			Layers[i]->CopyToDevice();
		}

		for(int i = 0; i < Weights.size(); ++i)
		{
			Weights[i]->CopyToDevice();
		}
	}

	void CopyToHost()
	{
		for(int i = 0; i < Layers.size(); ++i)
		{
			Layers[i]->CopyToHost();
		}

		for(int i = 0; i < Weights.size(); ++i)
		{
			Weights[i]->CopyToHost();
		}
	}

	void Print()
	{
		for(int y = 0; y < N; y++)
		{
			for(int x = 0; x < LAYERS; x++)
			{
				((Node)Layers[x]->HostData[y]).Print();
				cout << " ";

				int nextLayer = x + 1;
				if(nextLayer < LAYERS)
				{
					int block = Layers[nextLayer]->Length;
					for(int w = 0; w < block; w++)
					{
						int index = y * block + w;

						cout << y << "->" << w;

						((Element)Weights[x]->HostData[index]).Print();

						cout << " ";
					}
				}
			}

			cout << "\n";
		}
	}

	inline double randomValue()
	{
		return 0.5 + (double)rand() / RAND_MAX;
	}

	void randomValues(double* a, int n)
	{
		int i;
		for (i = 0; i < n; ++i)
		{
			a[i] = randomValue();
		}
	}

	void randomValues(Node* a, int n)
	{
		int i;
		for (i = 0; i < n; ++i)
		{
			a[i].Self.Value = randomValue();
			a[i].Bias.Value = randomValue();
		}
	}

	void randomValues(Element* a, int n)
	{
		int i;
		for (i = 0; i < n; ++i)
		{
			a[i].Value = randomValue();
		}
	}

	virtual ~NeuralNetwork()
	{
		for(int i = 0; i < Layers.size(); i++)
		{
			delete Layers[i];
		}

		for(int i = 0; i < Weights.size(); i++)
		{
			delete Weights[i];
		}
	}
};
