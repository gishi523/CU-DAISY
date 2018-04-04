#ifndef __GPU_MAX_ND_H__
#define __GPU_MAX_ND_H__

#include <opencv2/opencv.hpp>
#include <device_functions.h>
#include <host_defines.h>

template <typename T>
struct GpuMat3DPtr
{
	GpuMat3DPtr(T* data, int size1, int size2, int size3)
		: data(data), size1(size1), size2(size2), size3(size3) {}

	__device__ inline T& operator()(int i, int j, int k)
	{
		return *(this->data + (i * size2 + j) * size3 + k);
	}

	__device__ inline const T& operator()(int i, int j, int k) const
	{
		return *(this->data + (i * size2 + j) * size3 + k);
	}

	__device__ inline T* ptr(int i = 0, int j = 0, int k = 0)
	{
		return this->data + (i * size2 + j) * size3 + k;
	}

	__device__ inline const T* ptr(int i = 0, int j = 0, int k = 0) const
	{
		return this->data + (i * size2 + j) * size3 + k;
	}

	T* data;
	int size1, size2, size3;
};

template <typename T>
struct GpuMat4DPtr
{
	GpuMat4DPtr(T* data, int size1, int size2, int size3, int size4)
		: data(data), size1(size1), size2(size2), size3(size3), size4(size4) {}

	__device__ inline T& operator()(int i, int j, int k, int l)
	{
		return *(this->data + ((i * size2 + j) * size3 + k) * size4 + l);
	}

	__device__ inline const T& operator()(int i, int j, int k, int l) const
	{
		return *(this->data + ((i * size2 + j) * size3 + k) * size4 + l);
	}

	__device__ inline T* ptr(int i = 0, int j = 0, int k = 0, int l = 0)
	{
		return this->data + ((i * size2 + j) * size3 + k) * size4 + l;
	}

	__device__ inline const T* ptr(int i = 0, int j = 0, int k = 0, int l = 0) const
	{
		return this->data + ((i * size2 + j) * size3 + k) * size4 + l;
	}

	T* data;
	int size1, size2, size3, size4;
};

using GpuMat3DPtrf = GpuMat3DPtr<float>;
using GpuMat4DPtrf = GpuMat4DPtr<float>;

class GpuMat3D : public cv::cuda::GpuMat
{
public:
	using cv::cuda::GpuMat::GpuMat;

	GpuMat3D() {}

	GpuMat3D(int size1, int size2, int size3, int type)
	{
		create(size1, size2, size3, type);
	}

	GpuMat3D(int size1, int size2, int size3, int type, void* data)
	{
		create(size1, size2, size3, type, data);
	}

	void create(int size1, int size2, int size3, int type)
	{
		cv::cuda::GpuMat::GpuMat::create(size1 * size2 * size3, 1, type);
		this->size1 = size1;
		this->size2 = size2;
		this->size3 = size3;
	}

	void create(int size1, int size2, int size3, int type, void* data)
	{
		this->data = reinterpret_cast<uchar*>(data);
		this->size1 = size1;
		this->size2 = size2;
		this->size3 = size3;
	}

	template <typename T> inline T* ptr(int i = 0, int j = 0, int k = 0)
	{
		return (T*)this->data + (i * size2 + j) * size3 + k;
	}

	template <typename T> inline const T* ptr(int i = 0, int j = 0, int k = 0) const
	{
		return (T*)this->data + (i * size2 + j) * size3 + k;
	}

	template <typename T> operator GpuMat3DPtr<T>() const
	{
		return GpuMat3DPtr<T>((T*)data, size1, size2, size3);
	}

	int size1, size2, size3;
};

class GpuMat4D : public cv::cuda::GpuMat
{
public:
	using cv::cuda::GpuMat::GpuMat;

	GpuMat4D() {}

	GpuMat4D(int size1, int size2, int size3, int size4, int type)
	{
		create(size1, size2, size3, size4, type);
	}

	void create(int size1, int size2, int size3, int size4, int type)
	{
		cv::cuda::GpuMat::GpuMat::create(size1 * size2 * size3 * size4, 1, type);
		this->size1 = size1;
		this->size2 = size2;
		this->size3 = size3;
		this->size4 = size4;
	}

	template <typename T> inline T* ptr(int i = 0, int j = 0, int k = 0, int l = 0)
	{
		return (T*)this->data + ((i * size2 + j) * size3 + k) * size4 + l;
	}

	template <typename T> inline const T* ptr(int i = 0, int j = 0, int k = 0) const
	{
		return (T*)this->data + ((i * size2 + j) * size3 + k) * size4 + l;
	}

	template <typename T> operator GpuMat4DPtr<T>() const
	{
		return GpuMat4DPtr<T>((T*)data, size1, size2, size3, size4);
	}

	int size1, size2, size3, size4;
};

#endif // !__GPU_MAX_ND_H__
