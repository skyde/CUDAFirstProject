#include <stdio.h>
//using namespace std;

void random_ints(int* a, int n);

#define N (2048 * 2048)
#define M 512

__global__ void add(int *a, int *b, int *c, int n)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;

	if(index < n)
	{
		c[index] = a[index] + b[index];
	}
}

int main(void)
{
	int *a, *b, *c;
	int *d_a, *d_b, *d_c;
	int size = N * sizeof(int);

	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);

	a = (int *)malloc(size);
	b = (int *)malloc(size);
	c = (int *)malloc(size);

	random_ints(a, N);
	random_ints(b, N);

	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	add<<<((N + M - 1) / M), M>>>(d_a, d_b, d_c, N);

	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

	for(int i = 0; i < N && i < 512; i++)
	{
		printf ("%d + %d = %d \n", *(a + i), *(b + i), *(c + i));
	}

	free(a);
	free(b);
	free(c);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

//	cout << "test";
	return 0;
}

void random_ints(int* a, int n)
{
	int i;
	for (i = 0; i < n; ++i)
	{
		a[i] = rand() % 50;
	}
}
