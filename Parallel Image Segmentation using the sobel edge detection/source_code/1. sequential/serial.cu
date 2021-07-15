#include <cassert>
#include <cstdlib>
#include <stdio.h>
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
#define MASK_DIMENSION 3
#define OFFSET (MASK_DIMENSION / 2)
const char *iname = "lena_bw.pgm";
const char *Imp_type = "serial";

float* serial_convl(float* image,float *kernel, size_t width,int height) {
  float* output = (float*)malloc(sizeof(float)*width*height);
  for (int y = 0; y < width; ++y) {
                for (int x = 0; x < width; ++x) {
                        float val = 0;
                        for (int j = 0; j < MASK_DIMENSION; ++j) { //perform convolution on a pixel
                                for (int i = 0; i < MASK_DIMENSION; ++i) {
                                        int d_i = i-OFFSET;
                                        int d_j = j-OFFSET;
                                        if (d_i + x >= 0 && d_i + x < width && d_j + y >= 0 && d_j + y < width) { //check if in bounds of image
                                         val += (float)image[y*width+x+d_j*width+d_i] * kernel[j*MASK_DIMENSION+i];
                                        }
                                }
                        }
            if (val>0.4 || val<-0.4){
                output[y*width+x] = 1;
            }
            else{
                output[y*width+x] = 0;
            }
        }
    }
    return output;
 }

void runTest(int argc, char **argv);

int main(int argc, char **argv)
{
    printf("%s starting...\n", Imp_type);


    runTest(argc, argv);

    cudaDeviceReset();
    return 0;
}
void runTest(int argc, char **argv)
{

    //int devID = findCudaDevice(argc, (const char **) argv);

    //convulution mask
    //float *sharpening = (float*)malloc(sizeof(float)*3*3);
    float *edge_dect = (float*)malloc(sizeof(float)*3*3);
    //float *averaging = (float*)malloc(sizeof(float)*3*3);
    //float sharp[9] = {-1,-1,-1,-1,9,-1,-1,-1,-1};
    float edge[9] = {-1,0,1,-2,0,2,-1,0,1};
    //float av[9] = {1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9};
    //sharpening=&sharp[0];
    edge_dect=&edge[0];
    //averaging=&av[0];


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

      //unsigned int size = width * height * sizeof(float);
    printf("Loaded '%s', %d x %d pixels\n", iname, width, height);


    float *serial_data = NULL;

    StopWatchInterface *s_timer = NULL;
    sdkCreateTimer(&s_timer);
    sdkStartTimer(&s_timer);
    serial_data = serial_convl(h_d,edge_dect,width,height);
    sdkStopTimer(&s_timer);
    printf("Processing time for serial: %f (ms)\n", sdkGetTimerValue(&s_timer));
    printf("%.2f Mpixels/sec\n",(width *height / (sdkGetTimerValue(&s_timer) / 1000.0f)) / 1e6);
    sdkDeleteTimer(&s_timer);

     //Write output to file
    char serial_outputfile[1024];
    strcpy(serial_outputfile, img_p);
    strcpy(serial_outputfile + strlen(img_p) - 4, "_serial_out.pgm");
    sdkSavePGM(serial_outputfile, serial_data, width, height);
    printf("Wrote '%s'\n", serial_outputfile);
    free(img_p);
}