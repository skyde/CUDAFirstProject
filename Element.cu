#pragma once

#include <stdio.h>
#include <iostream>
#include <array>
#include <stdlib.h>
#include "Globals.h"

using namespace std;

struct __align__(sizeof(double) * 2) Element
{
	Element() : Value(0), Derivative(0)
	{
//		cout << "Element ctor";
	}

    double Value, Derivative;

	public:
	void Print()
	{
		cout << "[" << Value;
#if PRINT_DERIVATIVE
		cout << " " << Derivative;
#endif
		cout << "]";
	}
};
