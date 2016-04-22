#pragma once

#define PRINT_DERIVATIVE true
#define PRINT_PRECISION 6
#define PRINT_VERBOSE false

#define STEP_SIZE 0.001

#define M 1 // 512
#define LAYERS 4

// Total Threads
//#define N 3 // Nodes per layer
#ifndef NODES_IN_LAYER
#define NODES_IN_LAYER
namespace
{
	int NodesInLayer(int layer)
	{
		if(layer >= LAYERS - 1)
		{
			return 10;
		}

		return 768;
	}
}

#endif
// Block Size
