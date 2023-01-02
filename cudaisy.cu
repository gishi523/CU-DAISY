#include "cudaisy.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector_types.h>
#include <device_launch_parameters.h>

#include <opencv2/core/cuda_stream_accessor.hpp>

#include "gpu_mat_nd.h"

#define CUDA_CHECK(err) \
do {\
	if (err != cudaSuccess) {\
		printf("[CUDA Error] %s (code: %d) at %s:%d\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
	} \
} while (0)

#if CUDA_VERSION >= 9000
#define SHFL_XOR(var, delta) __shfl_xor_sync(0xffffffff, (var), (delta))
#else
#define SHFL_XOR(var, delta) __shfl_xor((var), (delta))
#endif

static const int REORDER_BLOCK_X = 32;
static const int REORDER_BLOCK_Y = 8;
static const int DESC_BLOCK_X = 32;
static const int DESC_BLOCK_Y = 8;
static const int UNROLL_NUM = 2;
static const int MAX_GRIDS = 256;
static __constant__ float cgridx[MAX_GRIDS];
static __constant__ float cgridy[MAX_GRIDS];

__global__ void reorderLayerKernel(GpuMat3DPtrf src, GpuMat3DPtrf dst)
{
	const int H = src.size1;
	const int N = src.size2 * src.size3;

	const int tx1 = threadIdx.x;
	const int ty1 = threadIdx.y;

	const int ty2 = (ty1 * blockDim.x + tx1) % H;
	const int tx2 = (ty1 * blockDim.x + tx1) / H;

	const int x1 = blockIdx.x * blockDim.x + tx1;
	const int x2 = blockIdx.x * blockDim.x + tx2;

	__shared__ float shsrc[REORDER_BLOCK_Y][REORDER_BLOCK_X];
	for (int y1 = ty1, y2 = ty2; y1 < H; y1 += blockDim.y, y2 += blockDim.y)
	{
		shsrc[ty1][tx1] = x1 < N ? src.data[y1 * N + x1] : 0.f;
		__syncthreads();
		if (x2 < N) dst.data[x2 * H + y2] = shsrc[ty2][tx2];
	}
}

__global__ void calc1stLayerKernel(cv::cuda::PtrStepf dx, cv::cuda::PtrStepf dy, cv::cuda::PtrStepf G,
	int cols, int rows, float cost, float sint)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= cols || y >= rows)
		return;

	G(y, x) = max(cost * dx(y, x) + sint * dy(y, x), 0.f);
}

__device__ inline void copy(const float* __restrict src, float* __restrict dst, int n)
{
	for (int i = threadIdx.x; i < n; i += blockDim.x)
		dst[i] = src[i];
}

__device__ inline void zero(float* dst, int n)
{
	for (int i = threadIdx.x; i < n; i += blockDim.x)
		dst[i] = 0;
}

__device__ inline float calcHistogramNI(float x, float y, GpuMat4DPtrf layers, int q, int h)
{
	const int rows = layers.size2;
	const int cols = layers.size3;
	const int ix = static_cast<int>(x + 0.5f);
	const int iy = static_cast<int>(y + 0.5f);

	if (ix < 0 || ix >= cols || iy < 0 || iy >= rows)
		return 0.f;

	return layers(q, iy, ix, h);
}

__device__ inline float calcHistogramBI(float x, float y, GpuMat4DPtrf layers, int q, int h)
{
	const int rows = layers.size2;
	const int cols = layers.size3;

	const int ix0 = static_cast<int>(x);
	const int iy0 = static_cast<int>(y);
	const int ix1 = ix0 + 1;
	const int iy1 = iy0 + 1;

	if (ix0 < 0 || ix1 >= cols || iy0 < 0 || iy1 >= rows)
		return 0.f;

	// A C --> pixel positions
	// B D
	const float* A = layers.ptr(q, iy0, ix0);
	const float* B = layers.ptr(q, iy1, ix0);
	const float* C = layers.ptr(q, iy0, ix1);
	const float* D = layers.ptr(q, iy1, ix1);

	const double dx0 = ix1 - x;
	const double dy0 = iy1 - y;
	const double dx1 = x - ix0;
	const double dy1 = y - iy0;

	const float w0 = static_cast<float>(dx0 * dy0);
	const float w1 = static_cast<float>(dx0 * dy1);
	const float w2 = static_cast<float>(dx1 * dy0);
	const float w3 = static_cast<float>(dx1 * dy1);

	return w0 * A[h] + w1 * B[h] + w2 * C[h] + w3 * D[h];
}

__device__ inline void normalize(const float* histogram, float* descriptor, int D)
{
	float sum = 0.f;
	for (int i = threadIdx.x; i < D; i += blockDim.x)
		sum += histogram[i] * histogram[i];

	for (int mask = 16; mask > 0; mask /= 2)
		sum += SHFL_XOR(sum, mask);

	const float scale = sum > 0.f ? rsqrtf(sum) : 1.f;
	for (int i = threadIdx.x; i < D; i += blockDim.x)
		descriptor[i] = scale * histogram[i];
}

__device__ inline void normalizePartial(const float* histogram, float* descriptor, float* scale, int S, int H)
{
	for (int i = threadIdx.x; i < S; i += blockDim.x)
	{
		float sum = 0.f;
		for (int h = i * H; h < i * H + H; h++)
			sum += histogram[h] * histogram[h];
		scale[i] = sum > 0.f ? rsqrtf(sum) : 1.f;
	}
	__syncthreads();

	for (int i = threadIdx.x; i < S * H; i += blockDim.x)
		descriptor[i] = scale[i / H] * histogram[i];
}

using CalcHistogram = float(*)(float, float, GpuMat4DPtrf, int, int);
template <CalcHistogram calcHistogram = calcHistogramNI>
__global__ void calcDescriptorsKernel(GpuMat4DPtrf layers, cv::cuda::PtrStepSzf descriptors, int Q, int T, int H, int norm)
{
	//const int rows = layers.size2;
	const int cols = layers.size3;
	const int S = Q * T + 1;
	const int D = S * H;
	const int d = UNROLL_NUM * (blockIdx.y * blockDim.y + threadIdx.y);

	extern __shared__ float cache[];
	float* histogram = cache + threadIdx.y * UNROLL_NUM * D;

	for (int i = threadIdx.x; i < D; i += blockDim.x)
	{
		const int q = max(i - H, 0) / (T * H);
		const int h = i % H;

#pragma unroll
		for (int k = 0; k < UNROLL_NUM; k++)
		{
			const int x = (d + k) % cols;
			const int y = (d + k) / cols;
			histogram[k * D + i] = calcHistogram(x + cgridx[i], y + cgridy[i], layers, q, h);
		}
	}

	if (norm == CUDAISY::NORM_NONE)
	{
#pragma unroll
		for (int k = 0; k < UNROLL_NUM && d + k < descriptors.rows; k++)
			copy(histogram + k * D, descriptors.ptr(d + k), D);
	}
	else if (norm == CUDAISY::NORM_FULL)
	{
#pragma unroll
		for (int k = 0; k < UNROLL_NUM && d + k < descriptors.rows; k++)
			normalize(histogram + k * D, descriptors.ptr(d + k), D);
	}
	else if (norm == CUDAISY::NORM_PARTIAL)
	{
		float* scale = cache + DESC_BLOCK_Y * UNROLL_NUM * D + threadIdx.y * UNROLL_NUM * S;

#pragma unroll
		for (int k = 0; k < UNROLL_NUM && d + k < descriptors.rows; k++)
			normalizePartial(histogram + k * D, descriptors.ptr(d + k), scale + k * S, S, H);
	}
}

static void calcCircularGrid(float R, int Q, int T, int H)
{
	std::vector<float> gridx(MAX_GRIDS), gridy(MAX_GRIDS);

	for (int i = 0; i < H; i++)
	{
		gridx[i] = 0.f;
		gridy[i] = 0.f;
	}

	int idx = H;
	for (int i = 0; i < Q; i++)
	{
		for (int j = 0; j < T; j++)
		{
			const double radius = R * (i + 1) / Q;
			const double theta = CV_2PI * j / T;
			const double dx = radius * cos(theta);
			const double dy = radius * sin(theta);
			for (int k = 0; k < H; k++)
			{
				gridx[idx] = static_cast<float>(dx);
				gridy[idx] = static_cast<float>(dy);
				idx++;
			}
		}
	}

	CUDA_CHECK(cudaMemcpyToSymbol(cgridx, gridx.data(), sizeof(float) * MAX_GRIDS));
	CUDA_CHECK(cudaMemcpyToSymbol(cgridy, gridy.data(), sizeof(float) * MAX_GRIDS));
}

static size_t calcSharedSize(int Q, int T, int H, int norm)
{
	const int S = Q * T + 1;
	const int D = S * H;

	size_t size = sizeof(float) * DESC_BLOCK_Y * UNROLL_NUM * D;
	if (norm == CUDAISY::NORM_PARTIAL)
		size += sizeof(float) * DESC_BLOCK_Y * UNROLL_NUM * S;

	int device;
	cudaDeviceProp prop;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&prop, device);

	CV_Assert(size <= prop.sharedMemPerBlock);

	return size;
}

void reorderLayer(const GpuMat3D& src, GpuMat3D& dst)
{
	const int rows = src.size2;
	const int cols = src.size3;
	const dim3 block(REORDER_BLOCK_X, REORDER_BLOCK_Y);
	const dim3 grid(cv::divUp(rows * cols, block.x), 1);
	reorderLayerKernel<<<grid, block>>>(src, dst);
	CUDA_CHECK(cudaGetLastError());
}

void calc1stLayer(const cv::cuda::GpuMat& dx, const cv::cuda::GpuMat& dy, cv::cuda::GpuMat& G,
	double cost, double sint, cv::cuda::Stream& _stream)
{
	const int rows = dx.rows;
	const int cols = dx.cols;

	G.create(rows, cols, CV_32F);

	const dim3 block(32, 8);
	const dim3 grid(cv::divUp(cols, block.x), cv::divUp(rows, block.y));
	cudaStream_t stream = cv::cuda::StreamAccessor::getStream(_stream);

	calc1stLayerKernel<<<grid, block, 0, stream>>>(dx, dy, G, cols, rows, static_cast<float>(cost), static_cast<float>(sint));
	CUDA_CHECK(cudaGetLastError());

	if (!stream)
		CUDA_CHECK(cudaDeviceSynchronize());
}

void calcDescriptors(const GpuMat4D& layers, cv::cuda::GpuMat& descriptors,
	float R, int Q, int T, int H, int norm, bool interpolation)
{
	CV_Assert(descriptors.cols <= MAX_GRIDS);

	calcCircularGrid(R, Q, T, H);

	const int rows = layers.size2;
	const int cols = layers.size3;

	const dim3 block(DESC_BLOCK_X, DESC_BLOCK_Y);
	const dim3 grid(1, cv::divUp(rows * cols / UNROLL_NUM, DESC_BLOCK_Y));
	const size_t shared = calcSharedSize(Q, T, H, norm);
	
	if (interpolation)
		calcDescriptorsKernel<calcHistogramBI><<<grid, block, shared>>>(layers, descriptors, Q, T, H, norm);
	else
		calcDescriptorsKernel<calcHistogramNI><<<grid, block, shared>>>(layers, descriptors, Q, T, H, norm);

	CUDA_CHECK(cudaGetLastError());
}
