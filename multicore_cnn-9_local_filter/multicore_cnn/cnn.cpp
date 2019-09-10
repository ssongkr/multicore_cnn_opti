#include "cnn.h"

extern const char* CLASS_NAME[];

double pooling_sec, conv_sec, conv1_sec, conv2_sec, conv3_sec, conv4_sec, conv5_sec, fc_sec, softmax_sec, find_max_sec, RELU_sec;

static void pooling2x2(float *input, float *output, int N) {
    int i, j, k, l;
    for (i = 0; i < N; i++) {
        for (j = 0; j < N; j++) {
            float max = 0;
            for (k = 0; k < 2; k++) {
                for (l = 0; l < 2; l++) {
                    float pixel = input[(i * 2 + k) * 2 * N + j * 2 + l];
                    max = (max > pixel) ? max : pixel;
                }
            }
            output[i * N + j] = max;
        }
    }
}

/*
 * D = channel size
 * N = width and height of an output image
 * Thus, input is (D, N * 2, N * 2) and output is (D, N, N).
 */
void pooling_layer(float *inputs, float *outputs, int D, int N) {
#ifdef PROFILE_ENABLE
	high_resolution_clock::time_point t1, t2;
	duration<double> time_span;
	t1 = high_resolution_clock::now();
#endif
	for (int i = 0; i < D; i++) {
        float * input = inputs + i * N * N * 4;
        float * output = outputs + i * N * N;
        pooling2x2(input, output, N);
    }
#ifdef PROFILE_ENABLE
	t2 = high_resolution_clock::now();
	time_span = duration_cast<duration<double>>(t2 - t1);
	pooling_sec += time_span.count();
#endif
}

/*
 * D2 = output channel size
 * D1 = input channel size
 * N = width and height of an input image
 * input image is zero-padded by 1.
 * Thus, input is (D1, N, N) and output is (D2, N, N)
 */
void convolution_layer(float *inputs, float *outputs, cl_mem filters, cl_mem biases, int D2, int D1, int N, int batch_size, int imageCnt) {
#ifdef PROFILE_ENABLE
	high_resolution_clock::time_point t1, t2;
	duration<double> time_span;
	t1 = high_resolution_clock::now();
#endif
	clConv(inputs, outputs, filters, biases, D2, D1, N, batch_size, imageCnt);
#ifdef PROFILE_ENABLE
	t2 = high_resolution_clock::now();
	time_span = duration_cast<duration<double>>(t2 - t1);
	conv_sec += time_span.count();
#endif
}

/*
 * M = output size
 * N = input size
 */
#define ReLU(x) (((x)>0)?(x):0)
static void fc_layer(float *input_neuron, float *output_neuron, float *weights, float *biases, int M, int N) {
#ifdef PROFILE_ENABLE
	high_resolution_clock::time_point t1, t2;
	duration<double> time_span;
	t1 = high_resolution_clock::now();
#endif
	int i, j;
    for (j = 0; j < M; j++) {
        float sum = 0;
        for (i = 0; i < N; i++) {
            sum += input_neuron[i] * weights[j * N + i];
        }
        sum += biases[j];
        output_neuron[j] = ReLU(sum);
    }
#ifdef PROFILE_ENABLE
	t2 = high_resolution_clock::now();
	time_span = duration_cast<duration<double>>(t2 - t1);
	fc_sec += time_span.count();
#endif
}

static void softmax(float *output, int N) {
#ifdef PROFILE_ENABLE
	high_resolution_clock::time_point t1, t2;
	duration<double> time_span;
	t1 = high_resolution_clock::now();
#endif
    int i;
    float max = output[0];
    for (i = 1; i < N; i++) {
        max = (output[i] > max)?output[i]:max;
    }
    float sum = 0;
    for (i = 0; i < N; i++) {
        sum += expf(output[i] - max);
    }
    for (i = 0; i < N; i++) {
        output[i] = expf(output[i] - max) / sum;
    }
#ifdef PROFILE_ENABLE
	t2 = high_resolution_clock::now();
	time_span = duration_cast<duration<double>>(t2 - t1);
	softmax_sec += time_span.count();
#endif
}

static int find_max(float *fc, int N) {
#ifdef PROFILE_ENABLE
	high_resolution_clock::time_point t1, t2;
	duration<double> time_span;
	t1 = high_resolution_clock::now();
#endif
    int i;
    int maxid = 0;
    float maxval = 0;
    for (i = 0; i < N; i++) {
        if (maxval < fc[i]) {
            maxval = fc[i];
            maxid = i;
        }
    }

#ifdef PROFILE_ENABLE
	t2 = high_resolution_clock::now();
	time_span = duration_cast<duration<double>>(t2 - t1);
	find_max_sec += time_span.count();
#endif
    return maxid;
}

float* alloc_layer(size_t n) {
    return (float*)malloc(n * sizeof(float));
}

void cnn_init() {
	int platform_idx = 0;
	int gpu_idx = 0;
	
	printf("platform_idx : ");
	scanf("%d", &platform_idx);
	printf("gpu_idx : ");
	scanf("%d", &gpu_idx);

	initOpenCL(platform_idx, gpu_idx);
}

cl_mem alloc_weight(float* filters, int D2, int D1);
cl_mem alloc_bias(float* bias, int D2);

void cnn(float *images, float **network, int *labels, float *confidences, int num_images, int batch_size) {
    // slice the network into weights and biases
    cl_mem w1_1, b1_1, w1_2, b1_2;
	cl_mem w2_1, b2_1, w2_2, b2_2;
	cl_mem w3_1, b3_1, w3_2, b3_2, w3_3, b3_3;
	cl_mem w4_1, b4_1, w4_2, b4_2, w4_3, b4_3;
	cl_mem w5_1, b5_1, w5_2, b5_2, w5_3, b5_3;
	float *w1, *b1, *w2, *b2, *w3, *b3;
    w1_1 = alloc_weight(network[0], 64, 3);     b1_1 = alloc_bias(network[1], 64);
    w1_2 = alloc_weight(network[2], 64, 64);    b1_2 = alloc_bias(network[3], 64);
    w2_1 = alloc_weight(network[4], 128, 64);   b2_1 = alloc_bias(network[5], 128);
    w2_2 = alloc_weight(network[6], 128, 128);  b2_2 = alloc_bias(network[7], 128);
    w3_1 = alloc_weight(network[8], 256, 128);  b3_1 = alloc_bias(network[9], 256);
    w3_2 = alloc_weight(network[10], 256, 256); b3_2 = alloc_bias(network[11], 256);
    w3_3 = alloc_weight(network[12], 256, 256); b3_3 = alloc_bias(network[13], 256);
    w4_1 = alloc_weight(network[14], 512, 256); b4_1 = alloc_bias(network[15], 512);
    w4_2 = alloc_weight(network[16], 512, 512); b4_2 = alloc_bias(network[17], 512);
    w4_3 = alloc_weight(network[18], 512, 512); b4_3 = alloc_bias(network[19], 512);
    w5_1 = alloc_weight(network[20], 512, 512); b5_1 = alloc_bias(network[21], 512);
    w5_2 = alloc_weight(network[22], 512, 512); b5_2 = alloc_bias(network[23], 512);
    w5_3 = alloc_weight(network[24], 512, 512); b5_3 = alloc_bias(network[25], 512);
    w1 = network[26]; b1 = network[27];
    w2 = network[28]; b2 = network[29];
    w3 = network[30]; b3 = network[31];

    // allocate memory for output of each layer
    float *c1_1, *c1_2, *p1;
    float *c2_1, *c2_2, *p2;
    float *c3_1, *c3_2, *c3_3, *p3;
    float *c4_1, *c4_2, *c4_3, *p4;
    float *c5_1, *c5_2, *c5_3, *p5;
    float *fc1, *fc2, *fc3;
    c1_1 = alloc_layer(64 * 32 * 32 * batch_size);
    c1_2 = alloc_layer(64 * 32 * 32 * batch_size);
    p1   = alloc_layer(64 * 16 * 16 * batch_size);
    c2_1 = alloc_layer(128 * 16 * 16 * batch_size);
    c2_2 = alloc_layer(128 * 16 * 16 * batch_size);
    p2   = alloc_layer(128 * 8 * 8 * batch_size);
    c3_1 = alloc_layer(256 * 8 * 8 * batch_size);
    c3_2 = alloc_layer(256 * 8 * 8 * batch_size);
    c3_3 = alloc_layer(256 * 8 * 8 * batch_size);
    p3   = alloc_layer(256 * 4 * 4 * batch_size);
    c4_1 = alloc_layer(512 * 4 * 4 * batch_size);
    c4_2 = alloc_layer(512 * 4 * 4 * batch_size);
    c4_3 = alloc_layer(512 * 4 * 4 * batch_size);
    p4   = alloc_layer(512 * 2 * 2 * batch_size);
    c5_1 = alloc_layer(512 * 2 * 2 * batch_size);
    c5_2 = alloc_layer(512 * 2 * 2 * batch_size);
    c5_3 = alloc_layer(512 * 2 * 2 * batch_size);
    p5   = alloc_layer(512 * 1 * 1 * batch_size);
    fc1  = alloc_layer(512 * batch_size);
    fc2  = alloc_layer(512 * batch_size);
    fc3  = alloc_layer(10 * batch_size);

	high_resolution_clock::time_point t1, t2;
	duration<double> time_span;

    // run network
    for(int i = 0; i < num_images; i+=batch_size)
    {
        float *image = images + i * 3 * 32 * 32;
		int imageCnt = batch_size;
		if (num_images - i < batch_size)
			imageCnt = num_images - i;

#ifdef PROFILE_ENABLE
		t1 = high_resolution_clock::now();
#endif
		convolution_layer(image, c1_1, w1_1, b1_1, 64, 3, 32, batch_size, imageCnt);
		convolution_layer(c1_1, c1_2, w1_2, b1_2, 64, 64, 32, batch_size, imageCnt);
#ifdef PROFILE_ENABLE
		t2 = high_resolution_clock::now();
		time_span = duration_cast<duration<double>>(t2 - t1);
		conv1_sec += time_span.count();
#endif
		for (int batch = 0; batch < imageCnt; batch++)
			pooling_layer(c1_2 + 64 * 32 * 32 * batch, p1 + 64 * 16 * 16 * batch, 64, 16);

#ifdef PROFILE_ENABLE
		t1 = high_resolution_clock::now();
#endif
		convolution_layer(p1, c2_1, w2_1, b2_1, 128, 64, 16, batch_size, imageCnt);
		convolution_layer(c2_1, c2_2, w2_2, b2_2, 128, 128, 16, batch_size, imageCnt);
#ifdef PROFILE_ENABLE
		t2 = high_resolution_clock::now();
		time_span = duration_cast<duration<double>>(t2 - t1);
		conv2_sec += time_span.count();
#endif
		for (int batch = 0; batch < imageCnt; batch++)
			pooling_layer(c2_2 + 128 * 16 * 16 * batch, p2 + 128 * 8 * 8 * batch, 128, 8);

#ifdef PROFILE_ENABLE
		t1 = high_resolution_clock::now();
#endif
		convolution_layer(p2, c3_1, w3_1, b3_1, 256, 128, 8, batch_size, imageCnt);
		convolution_layer(c3_1, c3_2, w3_2, b3_2, 256, 256, 8, batch_size, imageCnt);
		convolution_layer(c3_2, c3_3, w3_3, b3_3, 256, 256, 8, batch_size, imageCnt);
#ifdef PROFILE_ENABLE
		t2 = high_resolution_clock::now();
		time_span = duration_cast<duration<double>>(t2 - t1);
		conv3_sec += time_span.count();
#endif
		for (int batch = 0; batch < imageCnt; batch++)
			pooling_layer(c3_3 + 256 * 8 * 8 * batch, p3 + 256 * 4 * 4 * batch, 256, 4);

#ifdef PROFILE_ENABLE
		t1 = high_resolution_clock::now();
#endif
		convolution_layer(p3, c4_1, w4_1, b4_1, 512, 256, 4, batch_size, imageCnt);
		convolution_layer(c4_1, c4_2, w4_2, b4_2, 512, 512, 4, batch_size, imageCnt);
		convolution_layer(c4_2, c4_3, w4_3, b4_3, 512, 512, 4, batch_size, imageCnt);
#ifdef PROFILE_ENABLE
		t2 = high_resolution_clock::now();
		time_span = duration_cast<duration<double>>(t2 - t1);
		conv4_sec += time_span.count();
#endif
		for (int batch = 0; batch < imageCnt; batch++)
			pooling_layer(c4_3 + 512 * 4 * 4 * batch, p4 + 512 * 2 * 2 * batch, 512, 2);

#ifdef PROFILE_ENABLE
		t1 = high_resolution_clock::now();
#endif
		convolution_layer(p4, c5_1, w5_1, b5_1, 512, 512, 2, batch_size, imageCnt);
		convolution_layer(c5_1, c5_2, w5_2, b5_2, 512, 512, 2, batch_size, imageCnt);
		convolution_layer(c5_2, c5_3, w5_3, b5_3, 512, 512, 2, batch_size, imageCnt);
#ifdef PROFILE_ENABLE
		t2 = high_resolution_clock::now();
		time_span = duration_cast<duration<double>>(t2 - t1);
		conv5_sec += time_span.count();
#endif
		for (int batch = 0; batch < imageCnt; batch++)
		{
			pooling_layer(c5_3 + 512 * 2 * 2 * batch, p5 + 512 * 1 * 1 * batch, 512, 1);

			fc_layer(p5 + 512 * 1 * 1 * batch, fc1 + 512 * batch, w1, b1, 512, 512);
			fc_layer(fc1 + 512 * batch, fc2 + 512 * batch, w2, b2, 512, 512);
			fc_layer(fc2 + 512 * batch, fc3 + 10 * batch, w3, b3, 10, 512);

			softmax(fc3 + 10 * batch, 10);
			labels[i+batch] = find_max(fc3 + 10 * batch, 10);
			confidences[i+batch] = (fc3+10*batch)[labels[i+batch]];

#ifdef PROFILE_ENABLE
			fprintf(stdout, "Image %04d/%04d: %s %f\n", i + batch, num_images - 1, CLASS_NAME[labels[i + batch]], confidences[i + batch]);
#endif
		}
    }

    free(c1_1); free(c1_2); free(p1);
    free(c2_1); free(c2_2); free(p2);
    free(c3_1); free(c3_2); free(c3_3); free(p3);
    free(c4_1); free(c4_2); free(c4_3); free(p4);
    free(c5_1); free(c5_2); free(c5_3); free(p5);
    free(fc1); free(fc2); free(fc3);
}
