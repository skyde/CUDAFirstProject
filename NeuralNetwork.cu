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
using namespace std;

class NeuralNetwork
{
public:
	NeuralNetwork()
	{
		normal_distribution<double> distribution(0, 1.0);

		for(int i = 0; i < Layers.size(); i++)
		{
			cout << "Nodes in Layer " << i << " " << NodesInLayer(i) << "\n";
			Layers[i] = new SharedData<Node>(NodesInLayer(i));

			if(i != 0)
			{
				int length = Layers[i - 1]->Length * Layers[i]->Length;

				Weights[i - 1] = new SharedData<Element>(length);

				randomValues(
						Weights[i - 1]->HostData,
						Weights[i - 1]->Length,
						i);

//				cout << "weights " << i - 1 << ", l = " << length << "\n";
			}

			if(i > 0 && i < Layers.size() - 1)
			{
				randomValues(
						Layers[i]->HostData,
						Layers[i]->Length,
						i);
			}
		}
	}

	array<SharedData<Node>*, LAYERS> Layers;
	array<SharedData<Element>*, LAYERS - 1> Weights;

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

	double CaculateError(double* targets, bool print = false)
	{
		SharedData<Node>* outputs = Layers[Layers.size() - 1];

		double value = 0;

		for(int i = 0; i < outputs->Length; i++)
		{
			double target = outputs->HostData[i].Self.Value;

			double diff = abs(targets[i] - target);

			if(print)
			{
				cout << "(" << diff << ")";
			}

			value += diff * diff;
		}

		return value;
	}

	void PrintVerbose()
	{
		for(int y = 0; y < NodesInLayer(0); y++)
		{
			for(int x = 0; x < LAYERS; x++)
			{
				if(x >= NodesInLayer(y))
				{
					continue;
				}

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

	inline double randomValue(int layer)
	{
		return (distribution(generator) / NodesInLayer(layer)) * 0.001;
	}

//	void randomValues(double* a, int n)
//	{
//		int i;
//		for (i = 0; i < n; ++i)
//		{
//			a[i] = randomValue();
//		}
//	}

	void randomValues(Node* a, int n, int layer)
	{
		int i;
		for (i = 0; i < n; ++i)
		{
//			a[i].Self.Value = randomValue();
			a[i].Bias.Value = randomValue(layer);
		}
	}

	void randomValues(Element* a, int n, int layer)
	{
		int i;
		for (i = 0; i < n; ++i)
		{
			a[i].Value = randomValue(layer);
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
