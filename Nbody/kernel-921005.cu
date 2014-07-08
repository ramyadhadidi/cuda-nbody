#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "vector_types.h"

#include <Windows.h>
#include <GL/GL.h>
#include <GL/GLU.h>
#include <GL/glut.h>

#include <stdio.h>
#include <math.h>
#include <iostream>
#include <time.h>

using namespace std;

//###########################################################
cudaError_t addWithCuda(int *c, const int *a, const int *b, size_t size);
__global__ void addKernel(int *c, const int *a, const int *b);
//###########################################################
float3 bodyBodyInteractionCPU(float4 bi, float4 bj, float3 ai);
void printFloat3 (float3 a);
void printFloat4 (float4 a);

void GLkeyboard (unsigned char key, int x, int y);
void GLreshape (int w, int h);
void GLinit(void);
void GLdisplay(void);

const float G = 4.302e-3; // pc / M(sun) . (Km/s)^2
const float EPS2 = 1e-3;

static int year = 0, day = 0;


int main(int argc, char** argv)
{
		//###########################################################
		//###########################################################
	/*
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);
	*/
		//###########################################################
		//###########################################################

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

		float const timeSimulation = 0.1;
		float const timeStep = 1e-2;
		int const bodySize = 10;
		float4 b[bodySize]; //position + weight
		float3 v[bodySize]; //vel
		float3 a[bodySize]; //acc
		for (int i=0; i<bodySize; i++) {
			b[i].x = i;
			b[i].y = i;
			b[i].z = i;
			b[i].w = 1;

			v[i].x = 0;
			v[i].y = 0;
			v[i].z = 0;

			a[i].x = 0.0;
			a[i].y = 0.0;
			a[i].z = 0.0;
		}

		clock_t t0 = clock();
		cout << "Total Number of Steps " << (int)(timeSimulation/timeStep) << endl;
		for (int z=0; z<(int)(timeSimulation/timeStep); z++) {
			cout << "Iteration No: " << z << endl;
			for (int i=0; i<bodySize; i++) {
				cout << endl;
				cout << "body[" << i << "]" << endl;
				for (int j=0; j<bodySize; j++) {
					a[i] = bodyBodyInteractionCPU(b[i], b[j], a[i]);
				}
				v[i].x += a[i].x*timeStep;
				v[i].y += a[i].y*timeStep;
				v[i].z += a[i].z*timeStep;

				b[i].x += v[i].x*timeStep;
				b[i].y += v[i].y*timeStep;
				b[i].z += v[i].z*timeStep;
				printFloat4(b[i]);
			}
		}
		clock_t t1 = clock();
		cout << endl;
		cout << "Time: " << (t1-t0) << "us" << endl;
		cout << "Time: " << (t1-t0)/1000 << "ms" << endl;
		cout << "Time: " << (t1-t0)/1000000 << "s" << endl;
		//###########################################################
		//###########################################################

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
		cudaError_t cudaStatus;
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}


float3 bodyBodyInteractionCPU(float4 bi, float4 bj, float3 ai) {
	float3 r;
	r.x = bj.x - bi.x;
	r.y = bj.y - bi.y;
	r.z = bj.z - bi.z;

	float distSqr = (r.x*r.x) + (r.y*r.y) + (r.z*r.z) + EPS2;

	float distSixth = distSqr * distSqr * distSqr;
	float invDistSqr = 1.0f / sqrt(distSixth);

	float GM_r3 = G * bj.w * invDistSqr;

	ai.x += GM_r3 * r.x;
	ai.y += GM_r3 * r.y;
	ai.z += GM_r3 * r.z;

	return ai;
}

void printFloat3 (float3 a) {
	cout << "x= " << a.x << " ,y= " << a.y << " ,z= " << a.z << endl;
}

void printFloat4 (float4 a) {
	cout << "x= " << a.x << " ,y= " << a.y << " ,z= " << a.z << " ,w= " << a.w << endl;
}

void GLkeyboard (unsigned char key, int x, int y)
{
   switch (key) {
      case 'd':
         day = (day + 10) % 360;
         glutPostRedisplay();
         break;
      case 'D':
         day = (day - 10) % 360;
         glutPostRedisplay();
         break;
      case 'y':
         year = (year + 5) % 360;
         glutPostRedisplay();
         break;
      case 'Y':
         year = (year - 5) % 360;
         glutPostRedisplay();
         break;
      default:
         break;
   }
}

void GLreshape (int w, int h)
{
   glViewport (0, 0, (GLsizei) w, (GLsizei) h); 
   glMatrixMode (GL_PROJECTION);
   glLoadIdentity ();
   gluPerspective(60.0, (GLfloat) w/(GLfloat) h, 1.0, 20.0);
   glMatrixMode(GL_MODELVIEW);
   glLoadIdentity();
   gluLookAt (0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
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
   glutWireSphere(1.0, 20, 16);   /* draw sun */
   glRotatef ((GLfloat) year, 0.0, 1.0, 0.0);
   glTranslatef (2.0, 0.0, 0.0);
   glRotatef ((GLfloat) day, 0.0, 1.0, 0.0);
   glutWireSphere(0.2, 10, 8);    /* draw smaller planet */
   glPopMatrix();
   glutSwapBuffers();
}

//###########################################################
//###########################################################
// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, size_t size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

//###########################################################
//###########################################################