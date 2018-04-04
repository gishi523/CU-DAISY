/*********************************************************************
* Software License Agreement (BSD License)
*
* Copyright (c) 2009
* Engin Tola
* web : http://www.engintola.com
* email : engin.tola+libdaisy@gmail.com
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*********************************************************************/

/*
"DAISY: An Efficient Dense Descriptor Applied to Wide Baseline Stereo"
by Engin Tola, Vincent Lepetit and Pascal Fua. IEEE Transactions on
Pattern Analysis and achine Intelligence, 31 Mar. 2009.
IEEE computer Society Digital Library. IEEE Computer Society,
http:doi.ieeecomputersociety.org/10.1109/TPAMI.2009.77

"A fast local descriptor for dense matching" by Engin Tola, Vincent
Lepetit, and Pascal Fua. Intl. Conf. on Computer Vision and Pattern
Recognition, Alaska, USA, June 2008
*/

#include "cudaisy.h"
#include <cuda_runtime.h>cache

void calc1stLayer(const cv::cuda::GpuMat& dx, const cv::cuda::GpuMat& dy, cv::cuda::GpuMat& G,
	double cost, double sint, cv::cuda::Stream& stream = cv::cuda::Stream::Null());
void reorderLayer(const GpuMat3D& src, GpuMat3D& dst);
void calcDescriptors(const GpuMat4D& layers, cv::cuda::GpuMat& descriptors,
	float R, int Q, int T, int H, int norm, bool interpolation);

static cv::Size calcKernelSize(double sigma, double factor)
{
	int ksize = static_cast<int>(factor * sigma);
	if (ksize % 2 == 0)
		ksize++;
	ksize = std::max(ksize, 3);
	return cv::Size(ksize, ksize);
}

void CUDAISY::CalcOrientationLayers::initFilters(float R, int Q, int H)
{
	using namespace cv::cuda;
	const int border = cv::BORDER_REPLICATE;

	// for horizontal and vertical gradients
	if (!gauss0)
		gauss0 = createGaussianFilter(CV_32F, CV_32F, cv::Size(5, 5), 0.5, 0.5, border, border);
	if (!sobelx)
		sobelx = createSobelFilter(CV_32F, CV_32F, 1, 0, 1, 0.5, border, border);
	if (!sobely)
		sobely = createSobelFilter(CV_32F, CV_32F, 0, 1, 1, 0.5, border, border);

	// for first orientation maps
	const double sigma1 = sqrt(1.6 * 1.6 - 0.25);
	gauss1.resize(H);
	for (int i = 0; i < H; i++)
		if (!gauss1[i])
			gauss1[i] = createGaussianFilter(CV_32F, CV_32F, calcKernelSize(sigma1, 5), sigma1, sigma1, border, border);

	// for following orientation maps
	cv::Mat1d sigmas(Q, 1);
	for (int i = 0; i < Q; i++)
		sigmas(i) = R * (i + 1) / (2 * Q);

	gauss2.resize(Q);
	for (int i = 0; i < Q; i++)
	{
		const double sigma = i == 0 ? sigmas(0) : sqrt(sigmas(i) * sigmas(i) - sigmas(i - 1) * sigmas(i - 1));
		gauss2[i].resize(H);
		for (int j = 0; j < H; j++)
		{
			if (!gauss2[i][j])
				gauss2[i][j] = createGaussianFilter(CV_32F, CV_32F, calcKernelSize(sigma, 5), sigma, sigma, border, border);
		}
	}
}

void CUDAISY::CalcOrientationLayers::operator()(const cv::cuda::GpuMat& image, GpuMat4D& layers, float R, int Q, int H)
{
	initFilters(R, Q, H);
	buffers.create(Q + 1, H, image.rows, image.cols, CV_32F);
	layers.create(Q, image.rows, image.cols, H, CV_32F);

	// compute horizontal and vertical gradients
	image.convertTo(fimage, CV_32F, 1. / 255);
	gauss0->apply(fimage, blur);
	sobelx->apply(blur, dx);
	sobely->apply(blur, dy);

	// apply consecutive convolutions
	std::vector<cv::cuda::Stream> streams(H);
	tmps.resize(H);

	// compute first orientation maps
	for (int i = 0; i < H; i++)
	{
		cv::cuda::GpuMat G1(image.rows, image.cols, CV_32F, buffers.ptr<float>(0, i));
		const double theta = CV_2PI * i / H;
		calc1stLayer(dx, dy, tmps[i], cos(theta), sin(theta), streams[i]);
		gauss1[i]->apply(tmps[i], G1, streams[i]);
	}

	// consecutive convolutions
	for (int i = 0; i < Q; i++)
	{
		// cv::cuda's Gaussian filter shares a constant memory which contains kernel value.
		// Therefore, there needs synchronization before kernel value changes.
		cudaDeviceSynchronize();
		for (int j = 0; j < H; j++)
		{
			cv::cuda::GpuMat src(image.rows, image.cols, CV_32F, buffers.ptr<float>(i + 0, j));
			cv::cuda::GpuMat dst(image.rows, image.cols, CV_32F, buffers.ptr<float>(i + 1, j));
			gauss2[i][j]->apply(src, dst, streams[j]);
		}
	}

	// reorganizes the cube data so that histograms are sequential in memory
	for (int i = 0; i < Q; i++)
	{
		GpuMat3D buffer(H, image.rows, image.cols, CV_32F, buffers.ptr<float>(i + 1));
		GpuMat3D layer(image.rows, image.cols, H, CV_32F, layers.ptr<float>(i));
		reorderLayer(buffer, layer);
	}
}

CUDAISY::CUDAISY(const Parameters& param) : param_(param)
{
}

// full scope
void CUDAISY::compute(const cv::cuda::GpuMat& image, cv::cuda::GpuMat& descriptors)
{
	CV_Assert(image.type() == CV_8U);

	const float R = param_.R;
	const int Q = param_.Q;
	const int T = param_.T;
	const int H = param_.H;
	const int S = Q * T + 1;
	const int D = S * H;

	descriptors.create(image.rows * image.cols, D, CV_32F);
	calcOrientationLayers_(image, layers_, R, Q, H);
	calcDescriptors(layers_, descriptors, R, Q, T, H, param_.norm, param_.interpolation);
}