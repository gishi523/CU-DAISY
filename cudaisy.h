#ifndef __CUDAISY_H__
#define __CUDAISY_H__

#include <opencv2/core.hpp>

/** @brief DAISY class.

The class implements the DAISY descriptor
described in "DAISY: An Efficient Dense Descriptor Applied to Wide Baseline Stereo" by Engin Tola
*/
class CUDAISY
{
public:

	enum
	{
		NORM_NONE    = 100, //!< will not do any normalization
		NORM_PARTIAL = 101, //!< histograms are normalized independently for L2 norm equal to 1.0
		NORM_FULL    = 102, //!< descriptors are normalized for L2 norm equal to 1.0
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
		Parameters(float R = 15, int Q = 3, int T = 8, int H = 8, int norm = NORM_FULL, bool interpolation = true);
	};

	static cv::Ptr<CUDAISY> create(const Parameters& param = Parameters());

	/** @brief Calculates an DAISY descriptor
	@param image image to extract descriptors
	@param descriptors resulted descriptors array for all image pixels
	*/
	virtual void compute(cv::InputArray image, cv::OutputArray descriptors) = 0;

	virtual ~CUDAISY();
};

#endif // !__CUDAISY_H__
