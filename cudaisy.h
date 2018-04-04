#ifndef __CUDAISY_H__
#define __CUDAISY_H__

#include <opencv2/opencv.hpp>
#include "gpu_mat_nd.h"

/** @brief DAISY class.

The class implements the DAISY descriptor
described in "DAISY: An Efficient Dense Descriptor Applied to Wide Baseline Stereo" by Engin Tola
*/
class CUDAISY
{
public:

	enum
	{
		NORM_NONE = 100,    //!< will not do any normalization
		NORM_PARTIAL = 101, //!< histograms are normalized independently for L2 norm equal to 1.0
		NORM_FULL = 102,    //!< descriptors are normalized for L2 norm equal to 1.0
	};

	// S: number of histograms used in the descriptor = Q * T + 1
	// D: the total size of the descriptor vector = S * H
	struct Parameters
	{
		float R;            //!< distance from the center pixel to outer most grid point.
		int Q;              //!< number of convolved orientations layers with different sigmas.
		int T;              //!< number of histograms at a single layer.
		int H;              //!< number of bins in the histogram.
		int norm;           //!< descriptors normalization type
		bool interpolation; //!< switch to disable interpolation for speed improvement at minor quality loss

		// default settings
		Parameters(float R = 15, int Q = 3, int T = 8, int H = 8, int norm = NORM_FULL, bool interpolation = true)
			: R(R), Q(Q), T(T), H(H), norm(norm), interpolation(interpolation)
		{
		}
	};

	CUDAISY(const Parameters& param = Parameters());

	/** @brief Calculates an DAISY descriptor
	@param image image to extract descriptors
	@param descriptors resulted descriptors array for all image pixels
	*/
	void compute(const cv::cuda::GpuMat& image, cv::cuda::GpuMat& descriptors);

private:

	struct CalcOrientationLayers
	{
		using Filter = cv::Ptr<cv::cuda::Filter>;
		void initFilters(float R, int Q, int H);
		void operator()(const cv::cuda::GpuMat& image, GpuMat4D& layers, float R, int Q, int H);

		cv::cuda::GpuMat fimage, blur, dx, dy;
		std::vector<cv::cuda::GpuMat> tmps;
		GpuMat4D buffers;

		Filter sobelx, sobely, gauss0;
		std::vector<Filter> gauss1;
		std::vector<std::vector<Filter>> gauss2;
	};

	Parameters param_;
	GpuMat4D layers_;
	CalcOrientationLayers calcOrientationLayers_;
};

#endif // !__CUDAISY_H__
