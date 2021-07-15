#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <string>
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

#define tile_width 8
// Define the files that are to be save and the reference images for validation
const char *imageFilename = "lena_bw.pgm";

texture<float, 2, cudaReadModeElementType> tex;
texture<float,2,cudaReadModeElementType> tex_edge;


__global__ void conv_tex(float *out,int width,int height,int m_width,int m_height){
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;
    int offset = m_width/2;
    float value = 0;
    for (int j = 0; j < m_height; ++j) {
                for (int i = 0; i < m_width; ++i) {
                        int mapi = i-offset;
                        int mapj = j-offset;
                        float u = ((float)x + mapi - (float)width/2)/(float)(width);
                        float v = ((float)y + mapj - (float)width/2)/(float)(width);
                        value += tex2D(tex,y-(int)(m_width/2)+i , x-(int)(m_height/2)+j)*tex2D(tex_edge,i,j);
                }
            }
    if(value>0.4 || value<-0.4){
        out[x*width+y] = 1;
    }
    else{
        out[x*width+y] = 0;
    }

}

void runTest(int argc, char **argv);

int main(int argc, char **argv)
{
    runTest(argc, argv);
    cudaDeviceReset();
    return 0;
}

void runTest(int argc, char **argv)
{

    int devID = findCudaDevice(argc, (const char **) argv);

     //convulution mask
    float *edgeDectection = (float*)malloc(sizeof(float)*3*3);
    float edge[9] = {-1,0,1,-2,0,2,-1,0,1};
    edgeDectection=&edge[0];

    // load image from disk
    float *hData = NULL;
    unsigned int width, height;
    char *imagePath = sdkFindFilePath(imageFilename, argv[0]);

    if (imagePath == NULL)
    {
        printf("Unable to source image file: %s\n", imageFilename);
        exit(EXIT_FAILURE);
    }

    sdkLoadPGM(imagePath, &hData, &width, &height);

    unsigned int size = width * height * sizeof(float);

    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaChannelFormatDesc edge_cd = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaArray *cuArray;
    cudaArray *edge_cu;
    checkCudaErrors(cudaMallocArray(&cuArray,&channelDesc,width,height));
    checkCudaErrors(cudaMallocArray(&edge_cu,&edge_cd,3,3));
    checkCudaErrors(cudaMemcpyToArray(cuArray,0,0,hData,size,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToArray(edge_cu,0,0,edgeDectection,3*3*sizeof(float),cudaMemcpyHostToDevice));

    // Set texture parameters
    tex.addressMode[0] = cudaAddressModeWrap;
    tex.addressMode[1] = cudaAddressModeWrap;

    tex_edge.addressMode[0] = cudaAddressModeWrap;
    tex_edge.addressMode[1] = cudaAddressModeWrap;

    // Bind the array to the texture
    checkCudaErrors(cudaBindTextureToArray(tex, cuArray, channelDesc));
    checkCudaErrors(cudaBindTextureToArray(tex_edge, edge_cu, edge_cd));

    dim3 dimBlock(8, 8, 1);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y, 1);

    // tex_time
    float *txData = NULL;
    checkCudaErrors(cudaMalloc((void **) &txData, size));

    StopWatchInterface *t_timer = NULL;
    sdkCreateTimer(&t_timer);
    sdkStartTimer(&t_timer);

    conv_tex<<<dimGrid,dimBlock,0>>>(txData,width,height,3,3);

    getLastCudaError("Kernel execution failed");
    checkCudaErrors(cudaDeviceSynchronize());

    sdkStopTimer(&t_timer);
    printf("Processing time for texture: %f (ms)\n", sdkGetTimerValue(&t_timer));
    printf("%.2f Mpixels/sec\n",(width *height / (sdkGetTimerValue(&t_timer) / 1000.0f)) / 1e6);
    sdkDeleteTimer(&t_timer);

    // Allocate mem for the result on host side
    float *tex_out = (float *) malloc(size);
    checkCudaErrors(cudaMemcpy(tex_out,txData,size,cudaMemcpyDeviceToHost));

    char tex_outputfile[1024];
    strcpy(tex_outputfile, imagePath);
    strcpy(tex_outputfile + strlen(imagePath) - 4, "_texture_out.pgm");
    sdkSavePGM(tex_outputfile, tex_out, width, height);
    printf("Wrote '%s'\n", tex_outputfile);

    free(imagePath);

}
