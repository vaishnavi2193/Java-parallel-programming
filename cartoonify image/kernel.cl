#define GAUSSIAN_SUM 159.0
#define COLOUR_MASK (1 << COLOUR_BITS) - 1
#define COLOUR_BITS 8
#define RED 2
#define GREEN 1
#define BLUE 0
#define assert

//int colourValue(int pixel, int colour) {
//		return ((pixel >> (colour * COLOUR_BITS)) & COLOUR_MASK);
//}

int createPixel(int redValue, int greenValue, int blueValue) {
		return ((redValue << (2 * COLOUR_BITS)) + (greenValue << COLOUR_BITS) + blueValue);
	}

int wrap(int pos, int size) {
		if (pos < 0) {
			pos = -1 - pos;
		} else if (pos >= size) {
			pos = (size - 1) - (pos - size);
		}
		return pos;
	}
	
int* convolution(int xCentre, int yCentre, int* filter, int filterSize,int filterHalf,int width, int height, __global int* pixels) {
	int sum[3] ={0,0,0};	
	for (int filterY = 0; filterY < filterSize; filterY++) {
			int y = wrap(yCentre + filterY - filterHalf, height);
			for (int filterX = 0; filterX < filterSize; filterX++) {
				int x = wrap(xCentre + filterX - filterHalf, width);
				int rgb= pixels[y * width + x];
				int filterVal = filter[filterY * filterSize + filterX];
				
				//sum[RED]=colourValue(rgb, RED) * filterVal;
                                sum[RED]=((rgb >> (RED * COLOUR_BITS)) & COLOUR_MASK)*filterVal;
				//sum[GREEN]=colourValue(rgb, GREEN) * filterVal;
                                sum[GREEN]=((rgb >> (GREEN * COLOUR_BITS)) & COLOUR_MASK)*filterVal;
				//sum[BLUE]=colourValue(rgb, BLUE) * filterVal;
                                sum[BLUE]=((rgb >> (BLUE * COLOUR_BITS)) & COLOUR_MASK)*filterVal;
			}
		}
		return sum;
	}

__kernel void gaussianBlur(__global int *pixels,int width, int height, __global int *newPixels){
 
	int GAUSSIAN_FILTER[25] = {
		2,  4,  5,  4,  2, // sum=17
		4,  9, 12,  9,  4, // sum=38
		5, 12, 15, 12,  5, // sum=49
		4,  9, 12,  9,  4, // sum=38
		2,  4,  5,  4,  2  // sum=17
	};

	
	int x = get_global_id(0);
	int y = get_global_id(1);

	int* sum=convolution(x, y,GAUSSIAN_FILTER, 5,3,width, height,pixels);
	int red = clamp((int)((sum[RED]/GAUSSIAN_SUM)+ 0.5),0,255);
	int green = clamp((int)((sum[GREEN]/GAUSSIAN_SUM)+ 0.5),0,255);
	int blue = clamp((int)((sum[BLUE]/GAUSSIAN_SUM)+ 0.5),0,255);
	newPixels[y*width+x] = createPixel(red, green, blue);
}
