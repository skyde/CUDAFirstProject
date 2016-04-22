#pragma once
#include <stdio.h>
#include <iostream>
#include <array>
#include "helper_cuda.h"
#include <stdlib.h>
#include <random>

#include "SharedData.cu"
#include "Layer.cu"
#include "Element.cu"
#include "Node.cu"
#include "Globals.h"
//#include "NeuralNetworkDevice.cu"
using namespace std;


//template <class T>
class NeuralNetwork
{
public:
	NeuralNetwork()
	{
		normal_distribution<double> distribution(0, 1.0);
//		array<SharedData<Node>*, LAYERS> layers;
//		array<SharedData<Element>*, LAYERS - 1> weights;

		for(int i = 0; i < Layers.size(); i++)
		{
			Layers[i] = new SharedData<Node>(N);

			if(i != 0)
			{
				int length = Layers[i - 1]->Length * Layers[i]->Length;

				Weights[i - 1] = new SharedData<Element>(length);

				randomValues(
						Weights[i - 1]->HostData,
						Weights[i - 1]->Length);

//				cout << "weights " << i - 1 << ", l = " << length << "\n";
			}

			randomValues(
					Layers[i]->HostData,
					Layers[i]->Length);
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

	default_random_engine generator;
	normal_distribution<double> distribution;

	inline double randomValue()
	{
		return (distribution(generator) / N) * 0.01;
	}

//	void randomValues(double* a, int n)
//	{
//		int i;
//		for (i = 0; i < n; ++i)
//		{
//			a[i] = randomValue();
//		}
//	}

	void randomValues(Node* a, int n)
	{
		int i;
		for (i = 0; i < n; ++i)
		{
//			a[i].Self.Value = randomValue();
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
