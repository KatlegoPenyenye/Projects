
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#define MASK_DIMENSION 3
#define MASK_OFFSET (MASK_DIMENSION / 2)
#define tile_width 8
#define BLOCK_SIZE 8


const char *iname = "lena_bw.pgm";
const char *Imp_type = "simpleTexture";

//__constant__ float const_sharp[9] = {-1,-1,-1,-1,9,-1,-1,-1,-1};
__constant__ float const_edge[9] = {-1,0,1,-2,0,2,-1,0,1};
//__constant__ float const_av[9] = {1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9};

__global__ void convolution_shared(float *image, float *output, size_t width){
    __shared__ float image_ds[(tile_width + MASK_DIMENSION - 1)*(tile_width + MASK_DIMENSION - 1)];
        int s_width = (tile_width + MASK_DIMENSION - 1);
        int ty = threadIdx.y;
        int tx = threadIdx.x;
        int block_y = blockIdx.y;
        int block_x = blockIdx.x;
        int x = block_x * tile_width + tx;
        int y = block_y * tile_width + ty;
        int h_I_t = y - MASK_OFFSET;
        int h_I_b = y + MASK_OFFSET;
        int h_I_l = x - MASK_OFFSET;
        int h_I_r = x + MASK_OFFSET;

        if (h_I_t < 0 || h_I_l < 0)
                image_ds[ty*s_width+tx] = 0;
        else
                image_ds[ty*s_width+tx] = image[y*width+x - MASK_OFFSET*width - MASK_OFFSET];

        if (h_I_r >= width || h_I_t < 0)
                image_ds[ty*s_width+(tx+MASK_OFFSET+MASK_OFFSET)] = 0;
        else
                image_ds[ty*s_width+(tx+MASK_OFFSET+MASK_OFFSET)] = image[y*width+x - MASK_OFFSET*width + MASK_OFFSET];

        if (h_I_b >= width || h_I_l < 0)
                image_ds[(ty+MASK_OFFSET+MASK_OFFSET)*s_width+tx] = 0;
        else
                image_ds[(ty+MASK_OFFSET+MASK_OFFSET)*s_width+tx] = image[y*width+x + MASK_OFFSET*width - MASK_OFFSET];

        if (h_I_r >= width || h_I_b >= width)
                image_ds[(ty+MASK_OFFSET+MASK_OFFSET)*s_width+(tx+MASK_OFFSET+MASK_OFFSET)] = 0;
        else
                image_ds[(ty+MASK_OFFSET+MASK_OFFSET)*s_width+(tx+MASK_OFFSET+MASK_OFFSET)] = image[y*width+x + MASK_OFFSET*width + MASK_OFFSET];

        __syncthreads();

        float out = 0;
        for (int j = 0; j < MASK_DIMENSION; ++j) {
                for (int i = 0; i < MASK_DIMENSION; ++i) {
                        out += (float)image_ds[(j + ty)*s_width+(i + tx)] * const_edge[j*MASK_DIMENSION+i];
                }

        }
    if(out>0.4 || out<-0.4){
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
    //size_t shm_size = BLOCK_SIZE * sizeof(unsigned long long);
    convolution_shared<<<dimGrid,dimBlock,0>>>(d_d,g_d,width);

    getLastCudaError("Kernel execution failed");
    checkCudaErrors(cudaDeviceSynchronize());

    sdkStopTimer(&g_timer);
    printf("Processing time for shared: %f (ms)\n", sdkGetTimerValue(&g_timer));
    printf("%.2f Mpixels/sec\n",(width *height / (sdkGetTimerValue(&g_timer) / 1000.0f)) / 1e6);
    sdkDeleteTimer(&g_timer);
    // Allocate mem for the result on host side
    float *global_out = (float *) malloc(size);
    checkCudaErrors(cudaMemcpy(global_out,g_d,size,cudaMemcpyDeviceToHost));

    // Write result to file
    char gl_out[1024];
    strcpy(gl_out, img_p);
    strcpy(gl_out + strlen(img_p) - 4, "_share_out.pgm");
    sdkSavePGM(gl_out, global_out, width, height);
    printf("Wrote '%s'\n", gl_out);

    free(img_p);

    cudaDeviceReset();
    return 0;
}





