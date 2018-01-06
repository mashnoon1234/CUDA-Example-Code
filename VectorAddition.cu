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
      syncthreads();
    }
}

void __global__ VectorAdditionGPUSharedMemory(float* a, float* b, float* c, int size, int sharedMemorySize)
{

}

int main()
{
  int size = 10000;
  float start;
  float stop;
  float cpu;
  float gpu;

  // VectorAdditionCPU Start
  int* a, b, cCPU, cGPU;
  a = new int[size];
  b = new int[size];
  cCPU = new int[size];
  cGPU = new int[size];
  // Generating Vectors
  for(int i = 0; i < size; i++)
  {
    a[i] = i;
    b[i] = i;
  }
  start = clock();
  VectorAdditionCPU(a, b, cCPU, size);
  stop = clock();
  cpu = (stop - start) / (CLOCKS_PER_SEC) * 1000;
  cout << "Time needed for the CPU to add " << size << " pairs of integers : " << cpu << " ms" << endl;
  // VectorAdditionCPU End

  // VectorAdditionGPU Start
  int* device_a, device_b, device_c;
  cudaMalloc(&device_a, size * sizeof(int));
  cudaMalloc(&device_b, size * sizeof(int));
  cudaMalloc(&device_c, size * sizeof(int));
  start = clock();
  cudaMemcpy(device_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(device_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
  VectorAdditionGPU<<<size / 256 + 1, 256>>>(device_a, device_b, device_c, size);
  cudaMemcpy(cGPU, device_c, size * sizeof(int), cudaMemcpyDeviceToHost);
  stop = clock();
  gpu = (stop - start) / CLOCKS_PER_SEC * 1000;
  cout << "Time needed for the GPU to add " << size << " pairs of integers : " << gpu << " ms" << endl;
  // VectorAdditionGPU End

  cout << "Performance Gain using GPU : " << cpu / gpu << " times" << endl;

  delete [] a;
  delete [] b;
  delete [] c;
  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_c);

  return 0;
}
