#include <opencv2/opencv.hpp>
#include <chrono>
#include <map>
#include <cuda_runtime.h>
#include "cudaisy.h"

#ifdef WITH_OPENCV_DAISY
#include <opencv2/xfeatures2d.hpp>
#endif

void benchmark(const cv::Mat& image, float R = 15, int Q = 3, int T = 8, int H = 8,
	int norm = CUDAISY::NORM_FULL, bool interpolation = false, int iterations = 20)
{
	std::map<int, std::string> normStr =
	{
		{ CUDAISY::NORM_NONE, "NORM_NONE" }, { CUDAISY::NORM_PARTIAL, "NORM_PARTIAL" }, { CUDAISY::NORM_FULL, "NORM_FULL" }
	};

	std::cout << "====================================================================" << std::endl;
	std::cout << "Image Size      : " << image.size() << std::endl;
	std::cout << "Descriptor Size : " << (Q * T + 1) * H << std::endl;
	std::cout << "Normalize       : " << normStr[norm] << std::endl;
	std::cout << "Interpolation   : " << (interpolation ? "True" : "False") << std::endl;

	CUDAISY::Parameters(R, Q, T, H, norm, interpolation);
	CUDAISY daisyGPU(CUDAISY::Parameters(R, Q, T, H, norm, interpolation));

	cv::Mat descCPU, descGPU;
	cv::cuda::GpuMat d_image(image);
	cv::cuda::GpuMat d_desc;
	uint64_t sumGPU = 0;

#ifdef WITH_OPENCV_DAISY
	auto daisyCPU = cv::xfeatures2d::DAISY::create(R, Q, T, H, norm, cv::noArray(), interpolation, false);
	uint64_t sumCPU = 0;
#endif

	for (int i = 0; i <= iterations; i++)
	{
		const auto t0 = std::chrono::system_clock::now();

		daisyGPU.compute(d_image, d_desc);
		cudaDeviceSynchronize();

		const auto t1 = std::chrono::system_clock::now();
#ifdef WITH_OPENCV_DAISY
		daisyCPU->compute(image, descCPU);

		const auto t2 = std::chrono::system_clock::now();
#endif
		if (i > 0)
		{
			sumGPU += std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

#ifdef WITH_OPENCV_DAISY
			sumCPU += std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

			d_desc.download(descGPU);
			const double error = cv::norm(descCPU - descGPU);
			std::cout << "Iteration       : " << i << " ";
			std::cout << "(Error Per-Element: " << 100 * error / descCPU.size().area() << "[%])\r" << std::flush;
#else
			std::cout << "Iteration       : " << i << "\r";
#endif
		}
	}

	std::cout << std::endl;
	std::cout << "GPU-DAISY Time  : " << 1. * sumGPU / iterations << "[msec]" << std::endl;
#ifdef WITH_OPENCV_DAISY
	std::cout << "CPU-DAISY Time  : " << 1. * sumCPU / iterations << "[msec]" << std::endl;
	std::cout << "Speed Up        : " << 1. * sumCPU / sumGPU << "[x]" << std::endl;
#endif
	std::cout << "====================================================================" << std::endl << std::endl;
}

int main(int argc, char* argv[])
{
	if (argc < 2)
	{
		std::cout << "usage: " << argv[0] << " image" << std::endl;
		return 0;
	}

	cv::Mat image = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
	if (image.empty())
	{
		std::cout << "imread failed." << std::endl;
		return 0;
	}

	benchmark(image, 15, 3, 8, 8, CUDAISY::NORM_FULL, false);
	benchmark(image, 15, 3, 8, 8, CUDAISY::NORM_FULL, true);
	benchmark(image, 15, 3, 8, 8, CUDAISY::NORM_PARTIAL, false);
	benchmark(image, 15, 3, 8, 8, CUDAISY::NORM_PARTIAL, true);

	return 0;
}
