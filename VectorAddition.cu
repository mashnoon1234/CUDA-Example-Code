#include <iostream>
#include <iomanip>

using namespace std;

void VectorAdditionCPU(int* a, int* b, int* c, int size)
{
  for(int i = 0; i < size; i++)
    c[i] = a[i] + b[i];
}

void __global__ VectorAdditionGPU(int* a, int* b, int* c, int size)
{
  int index = threadIdx.x + blockIdx.x * blockDim.x;
	if(index < size)
	{
	    c[index] = a[index] + b[index];
	    __syncthreads();
	}
}

void __global__ VectorAdditionGPUSharedMemory(int* a, int* b, int* c, int size, int sharedMemorySize)
{
  // Yet to be done
}

int main()
{
  int size = 100000000;
  float start;
  float stop;
  float cpu;
  float gpu;

  // VectorAdditionCPU Start
  int *a, *b, *cCPU, *cGPU;
  a = new int[size];
  b = new int[size];
  cCPU = new int[size];
  cGPU = new int[size];
  // Generating Vectors
  for(int i = 0; i < size; i++)
  {
    a[i] = 100;
    b[i] = 200;
  }
  start = clock();
  VectorAdditionCPU(a, b, cCPU, size);
  stop = clock();
  cpu = (stop - start) / (CLOCKS_PER_SEC) * 1000;
  cout << "Time needed for the CPU to add " << size << " pairs of integers : " << cpu << " ms" << endl;
  // VectorAdditionCPU End

  // VectorAdditionGPU Start
  int *device_a, *device_b, *device_c;
  cudaMalloc(&device_a, size * sizeof(int));
  cudaMalloc(&device_b, size * sizeof(int));
  cudaMalloc(&device_c, size * sizeof(int));
  cudaMemcpy(device_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(device_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
  start = clock();
  VectorAdditionGPU<<<size / 1024 + 1, 1024>>>(device_a, device_b, device_c, size);
  stop = clock();
  cudaMemcpy(cGPU, device_c, size * sizeof(int), cudaMemcpyDeviceToHost);
  gpu = (stop - start) / (CLOCKS_PER_SEC) * 1000;
  cout << "Time needed for the GPU to add " << size << " pairs of integers : " << gpu << " ms" << endl;
  // VectorAdditionGPU End

  cout << "Performance Gain using GPU : " << (int)(cpu / gpu) << " times" << endl;

  bool error = false;
  for(int i = 0; i < size; i++)
  	if(cGPU[i] != cCPU[i])
  		error = true;

  if(error)
  	cout << "Results don't match!" << endl;

  delete [] a;
  delete [] b;
  delete [] cCPU;
  delete [] cGPU;
  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_c);

  return 0;
}
