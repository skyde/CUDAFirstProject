#pragma once
#include <stdio.h>
#include <iostream>
#include <array>
#include <stdlib.h>
#include "Element.cu"
#include "Globals.h"
using namespace std;

struct __align__(sizeof(Element) * 2) Node
{
	Node() : Self(), Bias()
	{

	}

	Element Self, Bias;

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
