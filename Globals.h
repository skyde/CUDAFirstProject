#pragma once

#define PRINT_DERIVATIVE true
#define PRINT_PRECISION 6
#define PRINT_VERBOSE false
#define PRINT_ERROR true

#define STEP_SIZE 0.001

#define M 1 // 512
#define LAYERS 4

#define MNIST_ELEMENT_SIZE 28 * 28
#define MNIST_ELEMENTS_TO_LOAD 100 // 1000

#define PRINT_MNIST_DATA false

// Total Threads
//#define N 3 // Nodes per layer
#define NODES_IN_LAST_LAYER 10
#ifndef NODES_IN_LAYER
#define NODES_IN_LAYER
namespace
{
	int NodesInLayer(int layer)
	{
		if(layer >= LAYERS - 1)
		{
			return NODES_IN_LAST_LAYER;
		}

		return 768;
	}
}

#endif
// Block Size
