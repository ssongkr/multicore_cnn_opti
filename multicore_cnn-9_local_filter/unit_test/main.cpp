int main() {}

//#include <stdio.h>
//#include <stdlib.h>
//#include <math.h>
//#include <string.h>
//#include <time.h>
//#include "cnn.h"
//
//static void convolution3x3(float *input, float *output, float *filter, int N) {
//	int i, j, k, l;
//	for (i = 0; i < N; i++) {
//		for (j = 0; j < N; j++) {
//			float sum = 0;
//			for (k = 0; k < 3; k++) {
//				for (l = 0; l < 3; l++) {
//					int x = i + k - 1;
//					int y = j + l - 1;
//					if (x >= 0 && x < N && y >= 0 && y < N)
//						sum += input[x * N + y] * filter[k * 3 + l];
//				}
//			}
//			output[i * N + j] += sum;
//		}
//	}
//}
//
///*
// * D2 = output channel size
// * D1 = input channel size
// * N = width and height of an input image
// * input image is zero-padded by 1.
// * Thus, input is (D1, N, N) and output is (D2, N, N)
// */
//#define ReLU(x) (((x)>0)?(x):0)
//static void convolution_layer_seq(float *inputs, float *outputs, float *filters, float *biases, int D2, int D1, int N) {
//	int i, j;
//
//	memset(outputs, 0, sizeof(float) * N * N * D2);
//
//	for (j = 0; j < D2; j++) {
//		for (i = 0; i < D1; i++) {
//			float * input = inputs + N * N * i;
//			float * output = outputs + N * N * j;
//			float * filter = filters + 3 * 3 * (j * D1 + i);
//			convolution3x3(input, output, filter, N);
//		}
//	}
//
//	for (i = 0; i < D2; i++) {
//		float * output = outputs + N * N * i;
//		float bias = biases[i];
//		for (j = 0; j < N * N; j++) {
//			output[j] = ReLU(output[j] + bias);
//		}
//	}
//}
//
//int main()
//{
//	int batch_size = 256;
//	clock_t start, end;
//
//	int num_images = batch_size;
//	float *images = read_images(num_images);
//	float *network = read_network();
//	float **network_sliced = slice_network(network);
//	int *labels = (int*)calloc(num_images, sizeof(int));
//	float *confidences = (float*)calloc(num_images, sizeof(float));
//
//	cnn_init();
//
//	float* c1_1 = alloc_layer(64 * 32 * 32 * batch_size);
//	float* c1_2 = alloc_layer(64 * 32 * 32 * batch_size);
//	float* w1_1 = network_sliced[0];
//	float* b1_1 = network_sliced[1];
//	float* w1_2 = network_sliced[2];
//	float* b1_2 = network_sliced[3];
//	float* p1 = alloc_layer(64 * 16 * 16 * batch_size);
//
//	float* clOut = alloc_layer(64 * 32 * 32 * batch_size);
//	float* seqOut = alloc_layer(64 * 32 * 32 * batch_size);
//
//	// conv test
//
//	convolution_layer(images, c1_1, w1_1, b1_1, 64, 3, 32, batch_size);
//	memcpy(clOut, c1_1, sizeof(float) * 64 * 32 * 32 * batch_size);
//
//	for (int batch = 0; batch < batch_size; batch++)
//	{
//		float* image = images + batch * 3 * 32 * 32;
//		convolution_layer_seq(image, c1_1, w1_1, b1_1, 64, 3, 32);
//		memcpy(seqOut + batch * 64 * 32 * 32, c1_1, sizeof(float) * 64 * 32 * 32);
//	}
//
//	if (memcmp(clOut, seqOut, sizeof(float) * 64 * 32 * 32 * batch_size) == 0)
//	{
//		printf("result same \n");
//	}
//	else
//	{
//		printf("result difference \n");
//		return -1;
//	}
//
//	// conv+pooling test
//
//	start = clock();
//	convolution_layer(images, c1_1, w1_1, b1_1, 64, 3, 32, batch_size);
//	convolution_layer(c1_1, c1_2, w1_2, b1_2, 64, 64, 32, batch_size);
//	for (int batch = 0; batch < batch_size; batch++)
//	{
//		pooling_layer(c1_2 + 64 * 32 * 32 * batch, p1 + 64 * 16 * 16 * batch, 64, 16);
//	}
//	memcpy(clOut, p1, sizeof(float) * 64 * 16 * 16 * batch_size);
//	end = clock();
//	printf("OpenCL Elapsed time: %f sec\n", (double)(end - start) / CLK_TCK);
//
//	start = clock();
//	for (int batch = 0; batch < batch_size; batch++)
//	{
//		float* image = images + batch * 3 * 32 * 32;
//		convolution_layer_seq(image, c1_1, w1_1, b1_1, 64, 3, 32);
//		convolution_layer_seq(c1_1, c1_2, w1_2, b1_2, 64, 64, 32);
//		pooling_layer(c1_2, p1, 64, 16);
//		memcpy(seqOut + batch * 64 * 16 * 16, p1, sizeof(float) * 64 * 16 * 16);
//	}
//	end = clock();
//	printf("seq    Elapsed time: %f sec\n", (double)(end - start) / CLK_TCK);
//
//	if (memcmp(clOut, seqOut, sizeof(float) * 64 * 16 * 16 * batch_size) == 0)
//	{
//		printf("result same \n");
//	}
//	else
//	{
//		printf("result difference \n");
//	}
//
//	for (int i = 0; i < 10; i++)
//	{
//		printf("%f ", seqOut[i]);
//	}
//
//	return 0;
//}