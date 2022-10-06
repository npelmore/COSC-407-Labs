/**************************************************************

 The program reads a BMP image file and creates a new
 image that is the negative or desaturated of the input file.

 **************************************************************/

#include "qdbmp.h"
#include <stdio.h>
#include <omp.h>

typedef enum {desaturate, negative} ImgProcessing ;

/* Creates a negative image of the input bitmap file */
int main() {
	const char* inFile = "okanagan.bmp";
	const char* outFile = "okanagan_processed.bmp";
	const ImgProcessing processingType = desaturate; //or negative
	int num_thread=16;

	UCHAR r, g, b;
	UINT width, height;
	UINT x, y;
	BMP* bmp;

	/* Read an image file */
	bmp = BMP_ReadFile(inFile);
	BMP_CHECK_ERROR(stdout, -1);

	/* Get image's dimensions */
	width = BMP_GetWidth(bmp);
	height = BMP_GetHeight(bmp);

	double t = omp_get_wtime();


#pragma omp parallel num_threads(num_thread) private(x,y,r,g,b)
	{
		int id = omp_get_thread_num();
		int sizeOfSection = width/num_thread;
		int start = id  * sizeOfSection;
		int end = start + sizeOfSection;

		/* Iterate through all the image's pixels */
		for (x = start; x < end; ++x) {



			for (y = 0; y < height; ++y) {
				/* Get pixel's RGB values */
				BMP_GetPixelRGB(bmp, x, y, &r, &g, &b);

				/* Write new RGB values */
				if(processingType == negative)
					BMP_SetPixelRGB(bmp, x, y, 255 - r, 255 - g, 255 - b);
				else if(processingType == desaturate){
					UCHAR gray = r * 0.3 + g * 0.59 + b * 0.11;
					BMP_SetPixelRGB(bmp, x, y, gray, gray, gray);
				}
			}

		}
	}


	/* calculate and print processing time*/
	t = 1000 * (omp_get_wtime() - t);
	printf("Finished image processing in %.1f ms.", t);

	/* Save result */
	BMP_WriteFile(bmp, outFile);
	BMP_CHECK_ERROR(stdout, -2);

	/* Free all memory allocated for the image */
	BMP_Free(bmp);

	return 0;
}
//Sequential: Finished image processing in 797.0 ms.
//2 Threads:Finished image processing in 632.0 ms.
//4 Threads:Finished image processing in 431.0 ms.
//8 Threads:Finished image processing in 384.0 ms.
//16 Threads:Finished image processing in 385.0 ms.
