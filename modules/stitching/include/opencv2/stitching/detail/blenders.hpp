/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_STITCHING_BLENDERS_HPP
#define OPENCV_STITCHING_BLENDERS_HPP

#if defined(NO)
#  warning Detected Apple 'NO' macro definition, it can cause build conflicts. Please, include this header before any Apple headers.
#endif

#include "opencv2/core.hpp"
#include "opencv2/core/cuda.hpp"

namespace cv {
namespace detail {

//! @addtogroup stitching_blend
//! @{

/** @brief Base class for all blenders.

Simple blender which puts one image over another
*/
class CV_EXPORTS Blender
{
public:
    virtual ~Blender() {}

    enum { NO, FEATHER, MULTI_BAND };
    static Ptr<Blender> createDefault(int type, bool try_gpu = false);

    /** @brief Prepares the blender for blending.

    @param corners Source images top-left corners
    @param sizes Source image sizes
     */
    void prepare(const std::vector<Point> &corners, const std::vector<Size> &sizes);
    /** @overload */
    virtual void prepare(Rect dst_roi);
    /** @brief Processes the image.

    @param img Source image
    @param mask Source image mask
    @param tl Source image top-left corners
     */
    virtual void feed(InputArray img, InputArray mask, Point tl);
    /** @brief Blends and returns the final pano.

    @param dst Final pano
    @param dst_mask Final pano mask
     */
    virtual void blend(InputOutputArray dst, InputOutputArray dst_mask);

protected:
    UMat dst_, dst_mask_;
    Rect dst_roi_;
};

/** @brief Simple blender which mixes images at its borders.
 */
class CV_EXPORTS FeatherBlender : public Blender
{
public:
    FeatherBlender(float sharpness = 0.02f);

    float sharpness() const { return sharpness_; }
    void setSharpness(float val) { sharpness_ = val; }

    void prepare(Rect dst_roi) CV_OVERRIDE;
    void feed(InputArray img, InputArray mask, Point tl) CV_OVERRIDE;
    void blend(InputOutputArray dst, InputOutputArray dst_mask) CV_OVERRIDE;

    //! Creates weight maps for fixed set of source images by their masks and top-left corners.
    //! Final image can be obtained by simple weighting of the source images.
    Rect createWeightMaps(const std::vector<UMat> &masks, const std::vector<Point> &corners,
                          std::vector<UMat> &weight_maps);

private:
    float sharpness_;
    UMat weight_map_;
    UMat dst_weight_map_;
};

inline FeatherBlender::FeatherBlender(float _sharpness) { setSharpness(_sharpness); }

/** @brief Blender which uses multi-band blending algorithm (see @cite BA83).
 */
class CV_EXPORTS MultiBandBlender : public Blender
{
public:
    MultiBandBlender(int try_gpu = false, int num_bands = 5, ElemType weight_type = CV_32FC1);

    int numBands() const { return actual_num_bands_; }
    void setNumBands(int val) { actual_num_bands_ = val; }

    void prepare(Rect dst_roi) CV_OVERRIDE;
    void feed(InputArray img, InputArray mask, Point tl) CV_OVERRIDE;
    void blend(InputOutputArray dst, InputOutputArray dst_mask) CV_OVERRIDE;

private:
    int actual_num_bands_, num_bands_;
    std::vector<UMat> dst_pyr_laplace_;
    std::vector<UMat> dst_band_weights_;
    Rect dst_roi_final_;
    bool can_use_gpu_;
    ElemType weight_type_; //CV_32F or CV_16S
#if defined(HAVE_OPENCV_CUDAARITHM) && defined(HAVE_OPENCV_CUDAWARPING)
    std::vector<cuda::GpuMat> gpu_dst_pyr_laplace_;
    std::vector<cuda::GpuMat> gpu_dst_band_weights_;
    std::vector<Point> gpu_tl_points_;
    std::vector<cuda::GpuMat> gpu_imgs_with_border_;
    std::vector<std::vector<cuda::GpuMat> > gpu_weight_pyr_gauss_vec_;
    std::vector<std::vector<cuda::GpuMat> > gpu_src_pyr_laplace_vec_;
    std::vector<std::vector<cuda::GpuMat> > gpu_ups_;
    cuda::GpuMat gpu_dst_mask_;
    cuda::GpuMat gpu_mask_;
    cuda::GpuMat gpu_img_;
    cuda::GpuMat gpu_weight_map_;
    cuda::GpuMat gpu_add_mask_;
    int gpu_feed_idx_;
    bool gpu_initialized_;
#endif
};


//////////////////////////////////////////////////////////////////////////////
// Auxiliary functions

void CV_EXPORTS normalizeUsingWeightMap(InputArray weight, InputOutputArray src);

void CV_EXPORTS createWeightMap(InputArray mask, float sharpness, InputOutputArray weight);

void CV_EXPORTS createLaplacePyr(InputArray img, int num_levels, std::vector<UMat>& pyr);
void CV_EXPORTS createLaplacePyrGpu(InputArray img, int num_levels, std::vector<UMat>& pyr);

// Restores source image
void CV_EXPORTS restoreImageFromLaplacePyr(std::vector<UMat>& pyr);
void CV_EXPORTS restoreImageFromLaplacePyrGpu(std::vector<UMat>& pyr);

//! @}

} // namespace detail
} // namespace cv

#endif // OPENCV_STITCHING_BLENDERS_HPP
