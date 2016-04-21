#pragma once
#include <stdio.h>
#include <iostream>
#include <array>
#include <stdlib.h>
using namespace std;
#include "Element.cu"

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

//		cout << "(" << Self.Value << " " << Self.Derivative << ")";
	}
};