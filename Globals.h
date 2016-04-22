#pragma once

#define PRINT_DERIVATIVE true
#define PRINT_PRECISION 6

#define M 1 // 512

#define LAYERS 2

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
			return 1;
		}

		return 3;
	}
}

#endif
// Block Size
