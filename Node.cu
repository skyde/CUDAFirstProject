#pragma once
#include <stdio.h>
#include <iostream>
#include <array>
#include <stdlib.h>
#include "Element.cu"
#include "Globals.h"
using namespace std;

enum ActivationStyle
{
	ActivationNothing = 0,
	ActivationTanH = 1
};

struct __align__(sizeof(Element)) Node
{
	Node() : Self(), Bias(), Activation()
	{

	}

	Element Self, Bias;
	ActivationStyle Activation;

	public:
	void Print()
	{
		cout << "(" << Self.Value;
#if PRINT_DERIVATIVE
		cout << " " << Self.Derivative;
#endif
		cout << ")";

		cout << "{" << Bias.Value;
#if PRINT_DERIVATIVE
		cout << " " << Bias.Derivative;
#endif
		cout << "}";

//		cout << "(" << Self.Value << " " << Self.Derivative << ")";
	}
};
