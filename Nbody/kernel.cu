#ifndef __CUDACC__  
    #define __CUDACC__
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vector_types.h"
#include <helper_cuda.h>				//must add C:\ProgramData\NVIDIA Corporation\CUDA Samples\v5.0\common\inc to 
																//"CUDA C++" -> "common" -> "additional include directories"

#include <Windows.h>
#include <GL/GL.h>
#include <GL/GLU.h>
#include <GL/glut.h>

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
void readConfigfile(string ConfigFileName);
void readCSV(string CSVFileName);
void printFloat3 (float3 a);
void printFloat4 (float4 a);
void deviceQuery(void);
void printHelp(void);

void GLkeyboard (unsigned char key, int x, int y);
void GLkeyboardCUDA (unsigned char key, int x, int y);
void GLreshape (int w, int h);
void GLinit(void);
void GLdisplay(void);

__device__ float3 bodyBodyInteraction(float4 bi, float4 bj, float3 ai);
__device__ float3 tile_calculation(float4 myPosition, float3 accel);
__global__ void calculate_forces(float4* d_b, float3* d_a);
__global__ void calculate_positions(float4* d_b, float3* d_v, float3* d_a);

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

float lookatz = 1.0;
float lookaty = 0.0;
float lookatx = 0.0;

int main(int argc, char* argv[])
{
	
	//==========================================
	//Arg processing

	if (argc < 2) {		
		cout << "Please define a config file" << endl;
		cout << "\tor" << endl;
		cout << "For help use -h" << endl;
		exit(1);
	}

	if (argc == 2) {
		if (strcmp(argv[1],"-h") == 0) { printHelp(); return 0; }
	}

	deviceQuery();

	configFile = string (argv[1]);
	outputFile = false;
	graphics = false;
	CPU = false;
	GPU = false;

	for (int i=2; i< argc; i++) {
		cout << argv[i] << endl;
		if (strcmp(argv[i],"-w") == 0) { outputFile = true; }
		else if (strcmp(argv[i],"-g") == 0) { graphics = true; }
		else if (strcmp(argv[i], "-cpu" ) == 0) { CPU = true; }
		else if (strcmp(argv[i], "-gpu" ) == 0) { GPU = true; }
		else { cout << "Invalid argument" << endl; exit(1); }
	}

	//Arg processing
	//==========================================
	//Begin

	readConfigfile(configFile);

	b = new float4[bodySize]; //position + weight
	v = new float3[bodySize]; //vel
	a = new float3[bodySize]; //acc

	//Begin
	//==========================================
	//bodies properties get
	if (randomAssignment == "no") {
		readCSV(CSVFileName);
	}
	else if (randomAssignment =="yes") {
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
	}
	else {
		cout << "ERROR: Random Assignemnt valuse is invalid" << endl;
		exit(1);
	}
	//bodies properties get
	//==========================================
	//CPU part
	if (CPU) {
		if (graphics == true) {
			if (outputFile == true) {
			 outFile.open("output_CPU.txt", ios::out);
			 outFile << "Nbody Results" << endl << endl << endl;
			}
			glutInit(&argc, argv);
			glutInitDisplayMode (GLUT_DOUBLE | GLUT_RGB);
			glutInitWindowSize (800, 800); 
			glutInitWindowPosition (100, 100);
			glutCreateWindow (argv[0]);
			GLinit ();
			glutDisplayFunc(GLdisplay); 
			glutReshapeFunc(GLreshape);
			glutKeyboardFunc(GLkeyboard);
			glutMainLoop();
			//Notice: The program never retruns from glutloop
		}

		else if (graphics == false) {
			if (outputFile == true) {
			 outFile.open("output_CPU.txt", ios::out);
			 outFile << "Nbody Results" << endl << endl << endl;
			}

			clock_t t0_CPU = clock();
		
			int z=0;
			for (int k=0; k<totalIterationCPU; k++) {
						for (int i=0; i<bodySize; i++) {
							for (int j=0; j<bodySize; j++) {
								a[i] = bodyBodyInteractionCPU(b[i], b[j], a[i]);
							}

						v[i].x += a[i].x * timeStep;
						v[i].y += a[i].y * timeStep;
						v[i].z += a[i].z * timeStep;

						b[i].x += v[i].x * timeStep;
						b[i].y += v[i].y * timeStep;
						b[i].z += v[i].z * timeStep;

						a[i].x = 0.0;
						a[i].y = 0.0;
						a[i].z = 0.0;
					}
					if (outputFile == true) {
						z++;
						if (z==stepsToWriteFile) {
							z=0;
							for (int i=0; i<bodySize; i++) {
								outFile << "b " << i << ": x= " << setw(6) << b[i].x << " ,y= " << setw(6) << b[i].y << " ,z= " << setw(6) << b[i].z <<  endl;
								outFile << "v " << i << ": x= " << setw(6) << v[i].x << " ,y= " << setw(6) << v[i].y << " ,z= " << setw(6) << v[i].z <<  endl;
								outFile << endl;	
							}
						}
					}
				}

			clock_t t1_CPU = clock();
			cout <<	"LOG: CPU Time for " << totalIterationCPU << " iteration: " << (t1_CPU-t0_CPU) / (double) CLOCKS_PER_SEC << " s" <<  endl;

			if (outputFile == true) 
				outFile.close();
		}
	}
	//CPU part
	//==========================================
	//GPU part
	if (GPU) {
		if (graphics == true) {

			if (outputFile == true) {
			 outFile.open("output_GPU.txt", ios::out);
			 outFile << "Nbody Results" << endl << endl << endl;
			}

			//CUDA Initialization for openGL rendering

			glutInit(&argc, argv);
			glutInitDisplayMode (GLUT_DOUBLE | GLUT_RGB);
			glutInitWindowSize (800, 800); 
			glutInitWindowPosition (100, 100);
			glutCreateWindow (argv[0]);
			GLinit ();
			glutDisplayFunc(GLdisplay); 
			glutReshapeFunc(GLreshape);
			glutKeyboardFunc(GLkeyboardCUDA);
			glutMainLoop();
			//Notice: The program never retruns from glutloop
		}
		//------graphics gpu end
		if (graphics == false) {
			if (outputFile == true) {
				 outFile.open("output_GPU.txt", ios::out);
				 outFile << "Nbody Results" << endl << endl << endl;
				}

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

			int i,j;
			//kernel invocation
			for (i=0, j=0; i<totalIterationGPU; i++) {
				calculate_forces<<<gridSize, blockSize, sizeSharedMem>>> (d_b, d_a);
				calculate_positions<<<gridSize, blockSize>>> (d_b, d_v, d_a);

				if (outputFile == true) {
					j++;
					if (j == stepsToWriteFile) {
						j=0;
						cudaMemcpy(v, d_v, sizeCudaMem3, cudaMemcpyDeviceToHost);
						cudaMemcpy(b, d_b, sizeCudaMem4, cudaMemcpyDeviceToHost);
						for (int k=0; k<bodySize; k++) {
							outFile << "b " << k << ": x= " << setw(6) << b[k].x << " ,y= " << setw(6) << b[k].y << " ,z= " << setw(6) << b[k].z <<  endl;
							outFile << "v " << k << ": x= " << setw(6) << v[k].x << " ,y= " << setw(6) << v[k].y << " ,z= " << setw(6) << v[k].z <<  endl;
							outFile << endl;	
						}
					}
				}
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

			if (outputFile == true) 
					outFile.close();

		}
	}
	//GPU part
	//==========================================
	//Termination
  //Tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaError_t cudaStatus;
  cudaStatus = cudaDeviceReset();
  if (cudaStatus != cudaSuccess) {
      fprintf(stderr, "cudaDeviceReset failed!");
      return 1;
  }

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

float3 bodyBodyInteractionCPU(float4 bi, float4 bj, float3 ai) {
	//Total: [21FLOPS]
	float3 r;
	//[3FLOPS]
	r.x = bj.x - bi.x;
	r.y = bj.y - bi.y;
	r.z = bj.z - bi.z;

	//[6FLOPS]
	float distSqr = (r.x*r.x) + (r.y*r.y) + (r.z*r.z) + EPS2;	

	//[4FLOPS]
	float distSixth = distSqr * distSqr * distSqr;
	float invDistSqr = 1.0f / sqrt(distSixth);

	//[2FLOPS]
	float GM_r3 = G * bj.w * invDistSqr;	

	//[6FLOPS]
	ai.x += GM_r3 * r.x;
	ai.y += GM_r3 * r.y;
	ai.z += GM_r3 * r.z;

	return ai;
}

void printFloat3 (float3 a) {
	cout << "x= " << setw(6) << a.x << " ,y= " << setw(6) << a.y << " ,z= " << setw(6) << a.z << endl;
}

void printFloat4 (float4 a) {
	cout << "x= " << setw(6) << a.x << " ,y= " << setw(6) << a.y << " ,z= " << setw(6) << a.z << " ,w= " << setw(6) << a.w << endl;
}

void GLkeyboard (unsigned char key, int x, int y)
{
   switch (key) {
      case 'r':
				int z;
				for (int k=0, z=0; k<iterationBeforeGL; k++) {
					for (int i=0; i<bodySize; i++) {
						for (int j=0; j<bodySize; j++) {
							a[i] = bodyBodyInteractionCPU(b[i], b[j], a[i]);
						}
					v[i].x += a[i].x*timeStep;
					v[i].y += a[i].y*timeStep;
					v[i].z += a[i].z*timeStep;

					b[i].x += v[i].x*timeStep;
					b[i].y += v[i].y*timeStep;
					b[i].z += v[i].z*timeStep;

					a[i].x = 0.0;
					a[i].y = 0.0;
					a[i].z = 0.0;
				}
				if (outputFile == true) {
					z++;
					if (z==stepsToWriteFile) {
						z=0;
						for (int i=0; i<bodySize; i++) {
							outFile << "b " << i << ": x= " << setw(6) << b[i].x << " ,y= " << setw(6) << b[i].y << " ,z= " << setw(6) << b[i].z <<  endl;
							outFile << "v " << i << ": x= " << setw(6) << v[i].x << " ,y= " << setw(6) << v[i].y << " ,z= " << setw(6) << v[i].z <<  endl;
							outFile << endl;	
						}
					}
				}
			}
         glutPostRedisplay();
         break;
			case 'x':
				lookatz += 1;
				gluLookAt (0.0, 0.0, lookatz, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
				glutPostRedisplay();
				break;
			case 'q':
        if (outputFile == true) 
					outFile.close();
				exit(0);
      default:
         break;
   }
}

void GLkeyboardCUDA (unsigned char key, int x, int y)
{
   switch (key) {
      case 'r':
				 //Do something about it! 

				 //glutPostRedisplay();
         break;
			case 'x':
				lookatz += 1;
				gluLookAt (0.0, 0.0, lookatz, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
				glutPostRedisplay();
				break;
			case 'q':
				cudaFree(d_b);
				cudaFree(d_v);
				cudaFree(d_a);

        if (outputFile == true) 
					outFile.close();
				exit(0);
      default:
         break;
   }
}

void GLreshape (int w, int h)
{
   glViewport (0, 0, (GLsizei) w, (GLsizei) h); 
   glMatrixMode (GL_PROJECTION);
   glLoadIdentity ();
   gluPerspective(60.0, (GLfloat) w/(GLfloat) h, 1.0, 5000.0);
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();
}

void GLinit(void)
{
   glClearColor (0.0, 0.0, 0.0, 0.0);
   glShadeModel (GL_FLAT);
}

void GLdisplay(void)
{
   glClear (GL_COLOR_BUFFER_BIT);
   glColor3f (1.0, 1.0, 1.0);

   glPushMatrix();
	 for (int i=0; i<bodySize; i++) {
		 float3 a;
		 a.x = b[i].x /distanceReducingFactor;
		 a.y = b[i].y /distanceReducingFactor;
		 a.z = b[i].z /distanceReducingFactor;
		 //printFloat3(a);
		  glTranslatef (a.x, a.y, a.z);
			glutWireSphere(0.1, 10, 8);
			glTranslatef (-a.x, -a.y, -a.z);
	 }
   glPopMatrix();
   glutSwapBuffers();
}

void readConfigfile(string ConfigFileName) {
	int i=0;
	ifstream ConfigFile(ConfigFileName);

	if(ConfigFile == NULL) {
		cout << "ERROR: Cannot open Config File" << endl;
		exit(1);
	}

	string line;

	while (getline(ConfigFile,line)) {
		stringstream lineStream(line);
		string cell,cell_a;
		i++;

		getline(lineStream, cell, ':');

		if (cell == "*") {
			//this line is comment
		}

		else if (cell == "CSVFileName") {
			getline(lineStream, cell);
			CSVFileName = cell;
			cout << "LOG: CSV file name has set to " << CSVFileName << endl;
		}

		else if (cell == "RandomAssignment") {
			getline(lineStream, cell);
			randomAssignment = cell;
			cout << "LOG: Random Assignment has set to " << randomAssignment << endl;
			if (randomAssignment=="yes") {
				cout << "LOG: CSV file name will be ignored." << endl;
				cout << "LOG: Random Space Dimension will be ignored." << endl;
			}
			if (randomAssignment=="no") {
				cout << "LOG: All Body Weights will be ignored." << endl;
			}
		}

		else if (cell == "G") {
			getline(lineStream, cell);
			G = stof(cell);
			cout << "LOG: Gravitional constant has set to " << G << endl;
		}

		else if (cell == "BodySize") {
			getline(lineStream, cell);
			bodySize = stoi(cell);
			cout << "LOG: Total body number has set to " << bodySize << endl;
		}

		else if (cell == "TotalIterationCPU") {
			getline(lineStream, cell);
			float a = stof(cell);
			totalIterationCPU = int (a);
			cout << "LOG: CPU Iteration has set to " << totalIterationCPU << endl;
		}

		else if (cell == "TotalIterationGPU") {
			getline(lineStream, cell);
			float a = stof(cell);
			totalIterationGPU = int (a);
			cout << "LOG: GPU Iteration has set to " << totalIterationGPU << endl;
		}

		else if (cell == "IterationBeforeGL") {
			getline(lineStream, cell);
			iterationBeforeGL = stoi(cell);
			cout << "LOG: Itiration Before GL has set to " << iterationBeforeGL << endl;
		}

		else if (cell == "DistanceReducingFactor") {
			getline(lineStream, cell);
			distanceReducingFactor = stof(cell);
			cout << "LOG: Distance Reducing Factor has set to " << distanceReducingFactor << endl;
		}

		else if (cell == "CUDATileSize") {
			getline(lineStream, cell);
			tileSize = stoi(cell);
			cout << "LOG: CUDA Tile Size has set to " << tileSize << endl;
		}
		
		else if (cell == "StepsToWriteFile") {
			getline(lineStream, cell);
			stepsToWriteFile = stoi(cell);
			cout << "LOG: Steps To Write File Cuda has set to " << stepsToWriteFile << endl;
		}

		else if (cell == "TimeStep") {
			getline(lineStream, cell);
			timeStep = stof(cell);
			cout << "LOG: TimeStep has set to " << timeStep << endl;
		}

		else if (cell == "EPS2") {
			getline(lineStream, cell);
			EPS2 = stof(cell);
			cout << "LOG: Softenning factor has set to " << EPS2 << endl;
		}

		else if (cell == "RandomSpaceDimension") {
			getline(lineStream, cell);
			sphere_radius = stof(cell);
			cout << "LOG: Random Space Dimension set to " << sphere_radius << endl;
		}
	
		else if (cell == "AllBodyWeights") {
			getline(lineStream, cell);
			uniform_body_weight = stof(cell);
			cout << "LOG: All Body Weights set to " << uniform_body_weight << endl;
		}

		else {
			cout << "ERROR: Invalid Value in Config File in line " << i << endl;
			exit(1);
		}
	}
}

void readCSV(string CSVFileName) {
	ifstream CSVfile(CSVFileName);

	if (CSVfile == NULL) {
		cout << "ERROR: Cannot Open CSV file" << endl;
		exit(1);
	}

	cout << "LOG: Reading CSV file" << endl;
	string line;
	
	//Skip first row; they are descriptions
	getline(CSVfile,line);

	int i;
	for (i=0; i<bodySize; i++) {
		getline(CSVfile,line);
		stringstream lineStream(line);
		string cell;

		getline(lineStream, cell, ',');
		b[i].x = stof(cell);
		getline(lineStream, cell, ',');
		b[i].y = stof(cell);
		getline(lineStream, cell, ',');
		b[i].z = stof(cell);
		getline(lineStream, cell, ',');
		b[i].w = stof(cell);

		getline(lineStream, cell, ',');
		v[i].x = stof(cell);
		getline(lineStream, cell, ',');
		v[i].y = stof(cell);
		getline(lineStream, cell, ',');
		v[i].z = stof(cell);

		a[i].x = 0.0;
		a[i].y = 0.0;
		a[i].z = 0.0;

		/*cout << "LOG: CSV row " << i+1 << " has been saved with below details" << endl;
		cout <<"\tPositions    :"; printFloat4(b[i]);
		cout <<"\tVelocities   :"; printFloat3(v[i]);*/
		}
	cout << "LOG: CSV file has been saved (" << i+1 << " rows processed)" << endl;
}

void deviceQuery(void) {
		printf("\n\n----------------------------------------------------------------------\n\n");
	  printf("CUDA Device Query (Runtime API) version (CUDART static linking)\n\n");

    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        exit(EXIT_FAILURE);
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0)
    {
        printf("There are no available device(s) that support CUDA\n");
    }
    else
    {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }

    int dev, driverVersion = 0, runtimeVersion = 0;

    for (dev = 0; dev < deviceCount; ++dev)
    {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

        // Console log
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);
        printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);
        printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);

        char msg[256];
        sprintf(msg, "  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
                (float)deviceProp.totalGlobalMem/1048576.0f, (unsigned long long) deviceProp.totalGlobalMem);
        printf("%s", msg);

        printf("  (%2d) Multiprocessors x (%3d) CUDA Cores/MP:    %d CUDA Cores\n",
               deviceProp.multiProcessorCount,
               _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
               _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
        printf("  GPU Clock rate:                                %.0f MHz (%0.2f GHz)\n", deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);


#if CUDART_VERSION >= 5000
        // This is supported in CUDA 5.0 (runtime API device properties)
        printf("  Memory Clock rate:                             %.0f Mhz\n", deviceProp.memoryClockRate * 1e-3f);
        printf("  Memory Bus Width:                              %d-bit\n",   deviceProp.memoryBusWidth);

        if (deviceProp.l2CacheSize)
        {
            printf("  L2 Cache Size:                                 %d bytes\n", deviceProp.l2CacheSize);
        }
#else
        // This only available in CUDA 4.0-4.2 (but these were only exposed in the CUDA Driver API)
        int memoryClock;
        getCudaAttribute<int>(&memoryClock, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, dev);
        printf("  Memory Clock rate:                             %.0f Mhz\n", memoryClock * 1e-3f);
        int memBusWidth;
        getCudaAttribute<int>(&memBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, dev);
        printf("  Memory Bus Width:                              %d-bit\n", memBusWidth);
        int L2CacheSize;
        getCudaAttribute<int>(&L2CacheSize, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, dev);

        if (L2CacheSize)
        {
            printf("  L2 Cache Size:                                 %d bytes\n", L2CacheSize);
        }
#endif

        printf("  Max Texture Dimension Size (x,y,z)             1D=(%d), 2D=(%d,%d), 3D=(%d,%d,%d)\n",
               deviceProp.maxTexture1D   , deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
               deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
        printf("  Max Layered Texture Size (dim) x layers        1D=(%d) x %d, 2D=(%d,%d) x %d\n",
               deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1],
               deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1], deviceProp.maxTexture2DLayered[2]);

        printf("  Total amount of constant memory:               %lu bytes\n", deviceProp.totalConstMem);
        printf("  Total amount of shared memory per block:       %lu bytes\n", deviceProp.sharedMemPerBlock);
        printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
        printf("  Warp size:                                     %d\n", deviceProp.warpSize);
        printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
               deviceProp.maxThreadsDim[0],
               deviceProp.maxThreadsDim[1],
               deviceProp.maxThreadsDim[2]);
        printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
               deviceProp.maxGridSize[0],
               deviceProp.maxGridSize[1],
               deviceProp.maxGridSize[2]);
        printf("  Maximum memory pitch:                          %lu bytes\n", deviceProp.memPitch);
        printf("  Texture alignment:                             %lu bytes\n", deviceProp.textureAlignment);
        printf("  Concurrent copy and kernel execution:          %s with %d copy engine(s)\n", (deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
        printf("  Run time limit on kernels:                     %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
        printf("  Integrated GPU sharing Host Memory:            %s\n", deviceProp.integrated ? "Yes" : "No");
        printf("  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
        printf("  Alignment requirement for Surfaces:            %s\n", deviceProp.surfaceAlignment ? "Yes" : "No");
        printf("  Device has ECC support:                        %s\n", deviceProp.ECCEnabled ? "Enabled" : "Disabled");
#ifdef WIN32
        printf("  CUDA Device Driver Mode (TCC or WDDM):         %s\n", deviceProp.tccDriver ? "TCC (Tesla Compute Cluster Driver)" : "WDDM (Windows Display Driver Model)");
#endif
        printf("  Device supports Unified Addressing (UVA):      %s\n", deviceProp.unifiedAddressing ? "Yes" : "No");
        printf("  Device PCI Bus ID / PCI location ID:           %d / %d\n", deviceProp.pciBusID, deviceProp.pciDeviceID);

        const char *sComputeMode[] =
        {
            "Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)",
            "Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)",
            "Prohibited (no host thread can use ::cudaSetDevice() with this device)",
            "Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)",
            "Unknown",
            NULL
        };
        printf("  Compute Mode:\n");
        printf("     < %s >\n", sComputeMode[deviceProp.computeMode]);
    }

    // csv masterlog info
    // *****************************
    // exe and CUDA driver name
    printf("\n");
    std::string sProfileString = "deviceQuery, CUDA Driver = CUDART";
    char cTemp[16];

    // driver version
    sProfileString += ", CUDA Driver Version = ";
#ifdef WIN32
    sprintf_s(cTemp, 10, "%d.%d", driverVersion/1000, (driverVersion%100)/10);
#else
    sprintf(cTemp, "%d.%d", driverVersion/1000, (driverVersion%100)/10);
#endif
    sProfileString +=  cTemp;

    // Runtime version
    sProfileString += ", CUDA Runtime Version = ";
#ifdef WIN32
    sprintf_s(cTemp, 10, "%d.%d", runtimeVersion/1000, (runtimeVersion%100)/10);
#else
    sprintf(cTemp, "%d.%d", runtimeVersion/1000, (runtimeVersion%100)/10);
#endif
    sProfileString +=  cTemp;

    // Device count
    sProfileString += ", NumDevs = ";
#ifdef WIN32
    sprintf_s(cTemp, 10, "%d", deviceCount);
#else
    sprintf(cTemp, "%d", deviceCount);
#endif
    sProfileString += cTemp;

    // Print Out all device Names
    for (dev = 0; dev < deviceCount; ++dev)
    {
#ifdef _WIN32
    sprintf_s(cTemp, 13, ", Device%d = ", dev);
#else
    sprintf(cTemp, ", Device%d = ", dev);
#endif
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        sProfileString += cTemp;
        sProfileString += deviceProp.name;
    }

    sProfileString += "\n";
    printf("%s", sProfileString.c_str());

		printf("\n\n----------------------------------------------------------------------\n\n");
}

void printHelp(void) {
	cout << endl << "This program simulates nbody astronomical systems on CPU and GPU." << endl;
	cout << "Usage:\t.\\nbody [-h] config_file -gpu/cpu [-w] [-g]" << endl << endl;

	cout << setw(30) << "config_file :" << setw(50) << "A file containing parameters for simulation" << endl;
	cout << setw(30) << "-gpu/cpu :" << setw(50) << "Select executing platform" << endl;
	cout << setw(30) << "-w :" << setw(50) << "Enables Writing results in the output file" << endl;
	cout << setw(30) << "-g :" << setw(50) << "Enables Graphical output (only in cpu mode)" << endl;
	cout << endl;

	cout << "1. Config File: " << endl;
	cout << "Config file defines runnig parameters. Program reads and save the parametes based on their names ";
	cout << "after \":\". Below is the parameters names plus \":\" and their definitions. " << endl << endl;

	cout << setw(30) << "*: :" << setw(75) << "Shows a comment line" << endl;
	cout << setw(30) << "CSVFileName :" << setw(75) << "Set the name of CSV file for objects' positions & velocities" << endl;
	cout << setw(30) << "RandomAssignment: :" << setw(75) << "yes/no | Sets objects' positions & velocities randomely" << endl;
	cout << setw(30) << "G: :" << setw(75) << "gravitional constant" << endl;
	cout << setw(30) << "BodySize: :" << setw(75) << "Total number of bodies" << endl;
	cout << setw(30) << "TotalIterationCPU: :" << setw(75) << "Number of iterations in CPU mode" << endl;
	cout << setw(30) << "TotalIterationGPU: :" << setw(75) << "Number of iterations in GPU mode" << endl;
	cout << setw(30) << "DistanceReducingFactor: :" << setw(75) << "For better bodies representation in graphical output" << endl;
	cout << setw(30) << "TimeStep: :" << setw(75) << "Time step in runnig (constant during runnig)" << endl;
	cout << setw(30) << "EPS2: :" << setw(75) << "Softening factor" << endl;
	cout << setw(30) << "RandomSpaceDimension: :" << setw(75) << "If RandomAssignment set to yes this the random space dimentions (cubic)" << endl;
	cout << setw(30) << "AllBodyWeights: :" << setw(75) << "f RandomAssignment set to yes this is the weight of bodies (equal)" << endl;
	cout << setw(30) << "IterationBeforeGL: :" << setw(75) << "After how many iteration program updates graphical output" << endl;
	cout << setw(30) << "CUDATileSize: :" << setw(75) << "Tile size in CUDA runnig algorithm" << endl;
	cout << setw(30) << "StepsToWriteFile: :" << setw(75) << "After how many iteration program write results in output file" << endl;

	cout << endl;
	cout << "2. CSV file: " << endl;
	cout << "This file contains bodies positions, weights and velocities in zero time. The first row in csv file will be ignored. ";
	cout << "Only BodySize+1 rows will be read by the program.	The coloums are x/y/z/w/vx/vy/vz." << endl;

	cout << endl;
	cout << "3. Graphical Mode: " << endl;
	cout << "In graphical mode you can zoom out using X key and run the simulation by peressing R key." << endl;

	cout << endl << endl;
}