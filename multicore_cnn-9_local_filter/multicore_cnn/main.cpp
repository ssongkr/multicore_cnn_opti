#include "cnn.h"
#pragma warning(disable:4996)

int compare_result(int argc, char **argv);

extern double before_kernel_sec, profile_sec, pooling_sec, conv_sec, conv1_sec, conv2_sec, conv3_sec, conv4_sec, conv5_sec, fc_sec, softmax_sec, find_max_sec, RELU_sec;
extern long long write_nsec, kernel_nsec, read_nsec;
extern const char *CLASS_NAME[];

int main(int argc, char **argv)
{
    if (argc != 3) {
        print_usage_and_exit(argv);
    }

    int num_images = atoi(argv[1]);
	printf("num_images : ");
	scanf("%d", &num_images);

	int batch_size = 256;
	printf("batch_size : ");
	scanf("%d", &batch_size);

    float *images = read_images(num_images);
    float *network = read_network();
    float **network_sliced = slice_network(network);
    int *labels = (int*)calloc(num_images, sizeof(int));
    float *confidences = (float*)calloc(num_images, sizeof(float));

    cnn_init();
    clock_t start = clock();
    cnn(images, network_sliced, labels, confidences, num_images, batch_size);
	clock_t end = clock();
    printf("Elapsed time: %f sec\n", (double)(end - start) / CLK_TCK);

    FILE *of = fopen(argv[2], "w");
    int *labels_ans = read_labels(num_images);
    double acc = 0;
    for (int i = 0; i < num_images; ++i) {
        fprintf(of, "Image %04d: %s %f\n", i, CLASS_NAME[labels[i]], confidences[i]);
        if (labels[i] == labels_ans[i]) ++acc;
    }
    fprintf(of, "Accuracy: %f\n", acc / num_images);
    fclose(of);

    free(images);
    free(network);
    free(network_sliced);
    free(labels);
    free(confidences);
    free(labels_ans);

#ifdef PROFILE_ENABLE
	printf("  - conv     : %lf sec = (%.2lf + %.2lf + %.2lf + %.2lf + %.2lf) sec \n", conv_sec, conv1_sec, conv2_sec, conv3_sec, conv4_sec, conv5_sec);
	printf("    - before kernel : %lf sec \n", before_kernel_sec);
	printf("      - write       : %lf sec \n", write_nsec / 1000000000.0);
	printf("    - kernel        : %lf sec \n", kernel_nsec / 1000000000.0);
	printf("    - read          : %lf sec \n", read_nsec / 1000000000.0);
	printf("    - cl profile    : %lf sec \n", profile_sec);
	printf("    - RELU          : %lf sec \n", RELU_sec);
	printf("  - pooling  : %lf sec \n", pooling_sec);
	printf("  - fc       : %lf sec \n", fc_sec);
	printf("  - softmax  : %lf sec \n", softmax_sec);
	printf("  - find_max : %lf sec \n", find_max_sec);
#endif

	char* params[] = { "", "result.out", "seq.out", NULL };
	compare_result(3, params);

    return 0;
}
