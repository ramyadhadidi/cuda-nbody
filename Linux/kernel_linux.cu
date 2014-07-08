#ifndef __CUDACC__  
    #define __CUDACC__
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vector_types.h"
//#include <helper_cuda.h>				//must add C:\ProgramData\NVIDIA Corporation\CUDA Samples\v5.0\common\inc to 
																//"CUDA C++" -> "common" -> "additional include directories"

#include <stdio.h>
#include <math.h>
#include <iostream>
#include <memory>
#include <iomanip>
#include <time.h>
#include <sstream>
#include <fstream>
#include <string>
#include <cstring>

using namespace std;

float3 bodyBodyInteractionCPU(float4 bi, float4 bj, float3 ai);

__device__ float3 bodyBodyInteraction(float4 bi, float4 bj, float3 ai);
__device__ float3 tile_calculation(float4 myPosition, float3 accel);
__global__ void calculate_forces(float4* d_b, float3* d_a);
__global__ void calculate_positions(float4* d_b, float3* d_v, float3* d_a);

void deviceQuery(void);

string CSVFileName;						//CSVfile for objects properties
string configFile;						//Config File name
bool outputFile;							//Needed to save results?
ofstream outFile;							//if needed for output file
bool graphics;								//Graphical representation?
string randomAssignment;			//Do we need CSV file?
int bodySize;									//Total number of bodies
float EPS2;										//Softening Factor
float timeStep;								//Simulation timestep/accuracy
float G;											//Gravitional Constant
int sphere_radius;						//Random Space Dimension for creating bodies
float uniform_body_weight;		//For random generation body weights
float distanceReducingFactor;	//For conversation between distance unit in real in openGL
int iterationBeforeGL;				//How many itiration before dispalying
int totalIterationCPU;				//Number of the total iteration for CPU 
int	totalIterationGPU;				//Number of the total iteration for GPU
int stepsToWriteFile;					//After how many iteration write result to output file
int tileSize;									//tileSize for CUDA
bool CPU;											//input arg to run on CPU
bool GPU;											//input art to run on GPU

int sizeCudaMem3;							//required size for float3
int sizeCudaMem4;							//required size for float4
int sizeSharedMem;						//required size for shared memory CUDA
dim3 gridSize;								//gridSize global declration
dim3 blockSize;								//blockSize global declration

__constant__ float d_G[1];				//Gravitional Constant
__constant__ float d_EPS2[1];			//Softening Factor
__constant__ int d_bodySize[1];		//Total number of bodies
__constant__ int d_tileSize[1];		//tileSize for CUDA
__constant__ float d_timeStep[1];	//Simulation timestep/accuracy

float4 *b;										//position + weight
float3 *v;										//vel
float3 *a;										//acc

float4 *d_b;									//position + weight
float3 *d_v;									//vel
float3 *d_a;									//acc

int main(int argc, char* argv[])
{	
	randomAssignment = "yes";
	G = 3.3639e-20;
	totalIterationCPU = 5000;
	totalIterationGPU =  5000;
	iterationBeforeGL = 100;
	distanceReducingFactor = 1;
	stepsToWriteFile = 1;
	timeStep = 1;
	EPS2 = 1e-2;
	sphere_radius = 100;
	uniform_body_weight = 1;
	
	tileSize = 64;
	bodySize = 1538;
	
	cout << "LOG: BodySize= " << bodySize;
	cout << ", TileSize= " << tileSize << endl;

	b = new float4[bodySize]; //position + weight
	v = new float3[bodySize]; //vel
	a = new float3[bodySize]; //acc
	

	//Begin
	//==========================================
	//bodies properties get
		srand(time(0));

		for (int i=0; i<bodySize; i++) {
			b[i].x = rand()%2==1 ? rand() % sphere_radius : - rand() % sphere_radius;
			b[i].y = rand()%2==1 ? rand() % sphere_radius : - rand() % sphere_radius;
			b[i].z = rand()%2==1 ? rand() % sphere_radius : - rand() % sphere_radius;
			b[i].w = uniform_body_weight;

			v[i].x = 0;
			v[i].y = 0;
			v[i].z = 0;

			a[i].x = 0.0;
			a[i].y = 0.0;
			a[i].z = 0.0;
		}
		
	//==========================================
	//GPU part
		clock_t t0_GPU_with_data = clock();
		
		//Copy constant values - using constant memory
		cudaMemcpyToSymbol (d_G, &G, sizeof(float) * 1, 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol (d_EPS2, &EPS2, sizeof(float) * 1, 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol (d_bodySize, &bodySize, sizeof(int) * 1, 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol (d_tileSize, &tileSize, sizeof(int) * 1, 0, cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol (d_timeStep, &timeStep, sizeof(float) * 1, 0, cudaMemcpyHostToDevice);

		sizeCudaMem3 = sizeof(float3)*bodySize;
		sizeCudaMem4 = sizeof(float4)*bodySize;

		cudaMalloc((void**)&d_b, sizeCudaMem4);
		cudaMalloc((void**)&d_v, sizeCudaMem3);
		cudaMalloc((void**)&d_a, sizeCudaMem3);
		
		gridSize = ((bodySize+tileSize-1)/tileSize);
		blockSize = (tileSize);
		sizeSharedMem = sizeof(float4) * tileSize;

		cudaMemcpy(d_b, b, sizeCudaMem4, cudaMemcpyHostToDevice);
		cudaMemcpy(d_v, v, sizeCudaMem3, cudaMemcpyHostToDevice);
		
		clock_t t0_kernel = clock();

		int i;
		//kernel invocation
		for (i=0; i<totalIterationGPU; i++) {
			calculate_forces<<<gridSize, blockSize, sizeSharedMem>>> (d_b, d_a);
			calculate_positions<<<gridSize, blockSize>>> (d_b, d_v, d_a);
		}

		clock_t t1_kernel = clock();

		cudaMemcpy(v, d_v, sizeCudaMem3, cudaMemcpyDeviceToHost);
		cudaMemcpy(b, d_b, sizeCudaMem4, cudaMemcpyDeviceToHost);

		cudaFree(d_b);
		cudaFree(d_v);
		cudaFree(d_a);

		clock_t t1_GPU_with_data = clock();

		//performacne Report
		cout <<	"LOG: GPU Kernel + Data Transfer Time for " << totalIterationGPU << " iteration: " << (t1_GPU_with_data-t0_GPU_with_data) / (double) CLOCKS_PER_SEC << " s" <<  endl;
		cout <<	"LOG: GPU Kernel Time for " << totalIterationGPU << " itiration: " << (t1_kernel-t0_kernel) / (double) CLOCKS_PER_SEC << " s" <<  endl;


  return 0;
}

__global__
void calculate_positions(float4* d_b, float3* d_v, float3* d_a) {
	int body_id = blockIdx.x * blockDim.x + threadIdx.x;
	float4  myP = d_b[body_id];
	float3  myV = d_v[body_id];
	float3  myA = d_a[body_id];
	float timeStep = d_timeStep[0];

	myV.x += myA.x * timeStep;
	myV.y += myA.y * timeStep;
	myV.z += myA.z * timeStep;

	d_v[body_id] = myV;

	myP.x += myV.x * timeStep;
	myP.y += myV.y * timeStep;
	myP.z += myV.z * timeStep;

	d_b[body_id] = myP;
}

__global__
void calculate_forces (float4* d_b, float3* d_a) {	
	extern __shared__ float4 shPosition[];
	float4 myPosition;
	int i,tile;
	float3 acc = {0.0f, 0.0f, 0.0f};

	int body_id = blockIdx.x * blockDim.x + threadIdx.x;
	myPosition = d_b[body_id];

	for (i=0, tile=0; i<d_bodySize[0]; i+=d_tileSize[0], tile++) {
		int idx = tile * blockDim.x + threadIdx.x;
		shPosition[threadIdx.x] = d_b[idx];
		__syncthreads();
		acc = tile_calculation(myPosition, acc);
		__syncthreads();
	}

	//Save the result in global memory for the integration step
	d_a[body_id] = acc;
}

__device__
float3 tile_calculation(float4 myPosition, float3 accel) {
	int i;
	extern __shared__ float4 shPosition[];
	
	for(i=0; i<blockDim.x; i++) {
		accel = bodyBodyInteraction(myPosition, shPosition[i], accel);
	}

	return accel;
}

__device__
float3 bodyBodyInteraction(float4 bi, float4 bj, float3 ai) {
	//Total: [21FLOPS]
	float3 r;
	//[3FLOPS]
	r.x = bj.x - bi.x;
	r.y = bj.y - bi.y;
	r.z = bj.z - bi.z;

	//[6FLOPS]
	float distSqr = (r.x*r.x) + (r.y*r.y) + (r.z*r.z) + d_EPS2[0];	

	//[4FLOPS]
	float distSixth = distSqr * distSqr * distSqr;
	float invDistSqr = 1.0f / sqrtf(distSixth);

	//[2FLOPS]
	float GM_r3 = d_G[0] * bj.w * invDistSqr;	

	//[6FLOPS]
	ai.x += GM_r3 * r.x;
	ai.y += GM_r3 * r.y;
	ai.z += GM_r3 * r.z;

	return ai;
}