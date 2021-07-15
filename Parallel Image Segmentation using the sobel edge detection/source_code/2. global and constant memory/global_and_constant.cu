#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// Includes CUDA
#include <cuda_runtime.h>

// Utilities and timing functions
#include <helper_functions.h>    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include <helper_cuda.h>         // helper functions for CUDA error check

#define MAX_EPSILON_ERROR 5e-3f
#define MASK_DIMENSION 3
#define OFFSET (MASK_DIMENSION / 2)
#define BLOCK_SIZE 8
// Define the files that are to be save and the reference images for validation
//const char* filename = "lena_bw";
const char *iname = "lena_bw.pgm";
//const char *refFilename   = "ref_rotated.pgm";

const char *Imp_type = "Global_and_constant";


//__constant__ float cuda_kernel[MASK_DIMENSION * MASK_DIMENSION];
//__constant__ float const_sharp[9] = {-1,-1,-1,-1,9,-1,-1,-1,-1};
__constant__ float edge_kernel[9] = {-1,0,1,-2,0,2,-1,0,1};/////////change this
//__constant__ float const_av[9] = {1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9};

__global__ void __launch_bounds__(1024) convolution_global(float *image, float *output, size_t width){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        float val = 0;
        for (int j = 0; j < MASK_DIMENSION; ++j) {
                for (int i = 0; i < MASK_DIMENSION; ++i) {
                        int d_i = i-OFFSET;
                        int d_j = j-OFFSET;
                        if (d_i + x >= 0 && d_i + x < width && d_j + y >= 0 && d_j + y < width) { //check if in bounds $                                val += (float)image[y*width+x+d_j*width+d_i] * edge_kernel[j*MASK_DIMENSION+i];
                               val += (float)image[y*width+x+d_j*width+d_i] * edge_kernel[j*MASK_DIMENSION+i];

                        }
                }
        __syncthreads();
        }
        if(val>0.4 || val<-0.4){
        output[y*width+x] = 1;
    }
    else{
        output[y*width+x] = 0;
    }
}
int main(int argc, char **argv)
{
    printf("%s starting...\n", Imp_type);
    int devID = findCudaDevice(argc, (const char **) argv);

    // load image from disk
    float *h_d = NULL;
    unsigned int width, height;
    char *img_p = sdkFindFilePath(iname, argv[0]);

    if (img_p == NULL)
    {
        printf("Unable to source image file: %s\n", iname);
        exit(EXIT_FAILURE);
    }
    sdkLoadPGM(img_p, &h_d, &width, &height);

    unsigned int size = width * height * sizeof(float);
    printf("Loaded '%s', %d x %d pixels\n", iname, width, height);


    dim3 dimBlock(8, 8, 1);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);



    float *g_d = 0,*d_d=0;
    checkCudaErrors(cudaMalloc((void **) &g_d, size));
    checkCudaErrors(cudaMalloc((void **) &d_d, size));
    cudaMemcpy(d_d,h_d, size, cudaMemcpyHostToDevice);

    StopWatchInterface *g_timer = NULL;
    sdkCreateTimer(&g_timer);
    sdkStartTimer(&g_timer);

    convolution_global<<<dimGrid, dimBlock>>>(d_d,g_d,width);
    getLastCudaError("Kernel execution failed");
    checkCudaErrors(cudaDeviceSynchronize());

    sdkStopTimer(&g_timer);
    printf("Processing time for global: %f (ms)\n", sdkGetTimerValue(&g_timer));
    printf("%.2f Mpixels/sec\n",(width *height / (sdkGetTimerValue(&g_timer) / 1000.0f)) / 1e6);
    sdkDeleteTimer(&g_timer);

    // Allocate mem for the result on host side
    float *global_out = (float *) malloc(size);
    checkCudaErrors(cudaMemcpy(global_out,g_d,size,cudaMemcpyDeviceToHost));
    // Write result to file
    char gl_out[1024];
    strcpy(gl_out, img_p);
    strcpy(gl_out + strlen(img_p) - 4, "_global_out.pgm");
    sdkSavePGM(gl_out, global_out, width, height);
    printf("Wrote '%s'\n", gl_out);
    
    free(img_p);
    cudaDeviceReset();
    return 0;
}