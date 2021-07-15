#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#define MASK_DIM 3
#include "stb_image/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image/stb_image_write.h"


int main(int argc, char **argv)
{
    float kernel[9] = {-1,0,1,-2,0,2,-1,0,1};

    int p_s,p_s_id;
    int size,width,height,l_w,l_h,channel;
    double runtime;

    unsigned char *data = NULL;

    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&p_s_id);
    MPI_Comm_size(MPI_COMM_WORLD,&p_s);

    runtime = -MPI_Wtime();

    if(p_s_id==0){
        data = stbi_load("data/lena_bw.pgm", &width, &height, &channel,0);
        if(data == NULL){
            printf("unable to load data\n");
            exit(0);
        }
    }

    MPI_Bcast(&width, 1, MPI_INT, 0,MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0,MPI_COMM_WORLD);

    l_w = width/p_s;
    l_h = height/p_s;

    unsigned char *image = (unsigned char*)malloc(sizeof(unsigned char)*l_h*width);
    unsigned char *output = (unsigned char*)malloc(sizeof(unsigned char)*l_h*width);

    MPI_Scatter(data,l_h*width,MPI_UNSIGNED_CHAR,image,l_h*width,MPI_UNSIGNED_CHAR, 0,MPI_COMM_WORLD);
    ///Image segmentation part
    for (int y = 0; y < l_h; ++y) {
                for (int x = 0; x < width; ++x) {
                        int offset = MASK_DIM/2; //kernel offset with integer division
                        float value = 0;
                        for (int j = 0; j < MASK_DIM; ++j) { //perform convolution on a pixel
                                for (int i = 0; i < MASK_DIM; ++i) {
                                        int mapi = i-offset;
                                        int mapj = j-offset;
                                        if (mapi + x >= 0 && mapi + x < width && mapj + y >= 0 && mapj + y < width) { //check if in bounds of image
                                                value += ((float)image[y*width+x+mapj*width+mapi]/255) * kernel[j*MASK_DIM+i];
                                        }
                                }
                        }if(value>0.4 || value<-0.4){
                output[y*width+x] = 255;
            }else{
                output[y*width+x] = 0;
            }
        }
    }
    MPI_Gather(output, l_h*width,MPI_UNSIGNED_CHAR,data, l_h*width,MPI_UNSIGNED_CHAR, 0,MPI_COMM_WORLD);
    runtime += MPI_Wtime();
    if(p_s_id==0){
        stbi_write_bmp("data/lena_bw_mpi.bmp",width,height,channel,data);
        printf("wrote to data/lena_bw_mpi.bmp\n");
        printf("Total elapsed time: %10.6f\n", runtime);
        stbi_image_free(data);
    }
    MPI_Finalize();
    return 0;
}