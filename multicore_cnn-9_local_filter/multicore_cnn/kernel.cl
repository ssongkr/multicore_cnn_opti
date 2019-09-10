#define ReLU(x) (((x)>0)?(x):0)

__kernel void conv(
		__global float* inputs,
		__global float* filters,
		__global float* outputs,
		__constant float* biases,
		const int D1,
		const int D2,
		const int N,
		const int imageCnt,
		__local float* l_filter
	) 
{
	const int out_channel = get_global_id(0);
	const int batch = get_global_id(1) / (N*N);
	const int remain = get_global_id(1) % (N*N);
	const int i = remain / N;
	const int j = remain % N;
	const int lid = get_local_id(1);
	const int lsize = get_local_size(1);

    __global float* output = outputs + N * N * (D2*batch + out_channel);
	__global float* filter = filters + out_channel * D1 * 3 * 3;

	if (lid < D1)
	{
		for(int l=0;l<D1;l+=lsize)
			for(int k=0;k<9;k++)
				l_filter[(l+lid)*3*3 + k] = filter[(l+lid)*3*3 + k];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if (batch >= imageCnt)
		return;

	float sum = 0;
	for (int in_channel = 0; in_channel < D1; in_channel++)
    {
		__global float* input = inputs + N * N * (D1*batch + in_channel);
		//__global float* filter = filters + 3 * 3 * (out_channel * D1 + in_channel);

		for (int k = 0; k < 3; k++) {
			for (int l = 0; l < 3; l++) {
				int x = i + k - 1;
				int y = j + l - 1;
				if (x >= 0 && x < N && y >= 0 && y < N)
					sum += input[x * N + y] * l_filter[(in_channel*3*3) + (k*3) + l];
			}
		}
	}
	float bias = biases[out_channel];
	output[i * N + j] = ReLU(sum + bias);
}
