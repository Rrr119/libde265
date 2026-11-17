/*
 * H.265 video codec.
 * Copyright (c) 2026 Dirk Farin <dirk.farin@gmail.com>
 *
 * This file is part of libde265.
 *
 * libde265 is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation, either version 3 of
 * the License, or (at your option) any later version.
 *
 * libde265 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with libde265.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef DE265_SSE_HIGHBIT_H
#define DE265_SSE_HIGHBIT_H

#include <stddef.h>
#include <stdint.h>

// SSE4.1 kernels for high-bit-depth (9..16 bit) decoding.  The motion
// compensation and weighted-prediction functions are used for bit depths up to
// MC_MAX_BIT_DEPTH_INT16 (the int16_t intermediate representation); the intra
// prediction, deblocking and inverse-transform wrappers are safe for all
// 16-bit pixel depths.  Everything here is bit-identical to the scalar
// fallback implementation.

void put_unweighted_pred_16_sse4(uint16_t *dst, ptrdiff_t dststride,
                                 const int16_t *src, ptrdiff_t srcstride,
                                 int width, int height, int bit_depth);
void put_weighted_pred_avg_16_sse4(uint16_t *dst, ptrdiff_t dststride,
                                   const int16_t *src1, const int16_t *src2,
                                   ptrdiff_t srcstride, int width, int height,
                                   int bit_depth);
void put_weighted_pred_16_sse4(uint16_t *dst, ptrdiff_t dststride,
                               const int16_t *src, ptrdiff_t srcstride,
                               int width, int height,
                               int w, int o, int log2WD, int bit_depth);
void put_weighted_bipred_16_sse4(uint16_t *dst, ptrdiff_t dststride,
                                 const int16_t *src1, const int16_t *src2,
                                 ptrdiff_t srcstride, int width, int height,
                                 int w1, int o1, int w2, int o2, int log2WD,
                                 int bit_depth);

void put_hevc_epel_16_sse4(int16_t *dst, ptrdiff_t dststride,
                           const uint16_t *src, ptrdiff_t srcstride,
                           int width, int height,
                           int mx, int my, int16_t *mcbuffer, int bit_depth);
void put_hevc_epel_h_16_sse4(int16_t *dst, ptrdiff_t dststride,
                             const uint16_t *src, ptrdiff_t srcstride,
                             int width, int height,
                             int mx, int my, int16_t *mcbuffer, int bit_depth);
void put_hevc_epel_v_16_sse4(int16_t *dst, ptrdiff_t dststride,
                             const uint16_t *src, ptrdiff_t srcstride,
                             int width, int height,
                             int mx, int my, int16_t *mcbuffer, int bit_depth);
void put_hevc_epel_hv_16_sse4(int16_t *dst, ptrdiff_t dststride,
                              const uint16_t *src, ptrdiff_t srcstride,
                              int width, int height,
                              int mx, int my, int16_t *mcbuffer, int bit_depth);

// One wrapper per fractional position, matching put_hevc_qpel_16[x][y].
void put_hevc_qpel_16_sse4(int16_t *dst, ptrdiff_t dststride,
                           const uint16_t *src, ptrdiff_t srcstride,
                           int width, int height, int16_t *mcbuffer,
                           int xFrac, int yFrac, int bit_depth);

void intra_pred_dc_16_sse4(uint16_t *dst, ptrdiff_t stride, int nT, int cIdx,
                           const uint16_t *border);
void intra_pred_planar_16_sse4(uint16_t *dst, ptrdiff_t stride, int nT,
                               int cIdx, const uint16_t *border);
void intra_pred_angular_16_sse4(uint16_t *dst, ptrdiff_t stride, int bit_depth,
                                int disableBoundaryFilter, int xB0, int yB0,
                                int mode, int nT, int cIdx,
                                const uint16_t *border);

void deblock_luma_16_sse4(uint16_t *ptr, ptrdiff_t stride, int vertical,
                          int dE, int dEp, int dEq, int tc,
                          int filterP, int filterQ, int bitDepth);

// Wrappers for the acceleration table.  The inverse transforms accelerate
// 10-bit, which is the profile this path was added for; other high bit depths
// use the scalar implementation.
void transform_4x4_dst_add_16_sse4(uint16_t *dst, const int16_t *coeffs,
                                   ptrdiff_t stride, int bit_depth);
void transform_4x4_add_16_sse4(uint16_t *dst, const int16_t *coeffs,
                               ptrdiff_t stride, int bit_depth);
void transform_8x8_add_16_sse4(uint16_t *dst, const int16_t *coeffs,
                               ptrdiff_t stride, int bit_depth);
void transform_16x16_add_16_sse4(uint16_t *dst, const int16_t *coeffs,
                                 ptrdiff_t stride, int bit_depth);
void transform_32x32_add_16_sse4(uint16_t *dst, const int16_t *coeffs,
                                 ptrdiff_t stride, int bit_depth);

// Declarations for the per-position qpel wrappers used by sse.cc.
void put_hevc_qpel_0_0_16_sse4(int16_t*, ptrdiff_t, const uint16_t*, ptrdiff_t, int, int, int16_t*, int);
void put_hevc_qpel_0_1_16_sse4(int16_t*, ptrdiff_t, const uint16_t*, ptrdiff_t, int, int, int16_t*, int);
void put_hevc_qpel_0_2_16_sse4(int16_t*, ptrdiff_t, const uint16_t*, ptrdiff_t, int, int, int16_t*, int);
void put_hevc_qpel_0_3_16_sse4(int16_t*, ptrdiff_t, const uint16_t*, ptrdiff_t, int, int, int16_t*, int);
void put_hevc_qpel_1_0_16_sse4(int16_t*, ptrdiff_t, const uint16_t*, ptrdiff_t, int, int, int16_t*, int);
void put_hevc_qpel_1_1_16_sse4(int16_t*, ptrdiff_t, const uint16_t*, ptrdiff_t, int, int, int16_t*, int);
void put_hevc_qpel_1_2_16_sse4(int16_t*, ptrdiff_t, const uint16_t*, ptrdiff_t, int, int, int16_t*, int);
void put_hevc_qpel_1_3_16_sse4(int16_t*, ptrdiff_t, const uint16_t*, ptrdiff_t, int, int, int16_t*, int);
void put_hevc_qpel_2_0_16_sse4(int16_t*, ptrdiff_t, const uint16_t*, ptrdiff_t, int, int, int16_t*, int);
void put_hevc_qpel_2_1_16_sse4(int16_t*, ptrdiff_t, const uint16_t*, ptrdiff_t, int, int, int16_t*, int);
void put_hevc_qpel_2_2_16_sse4(int16_t*, ptrdiff_t, const uint16_t*, ptrdiff_t, int, int, int16_t*, int);
void put_hevc_qpel_2_3_16_sse4(int16_t*, ptrdiff_t, const uint16_t*, ptrdiff_t, int, int, int16_t*, int);
void put_hevc_qpel_3_0_16_sse4(int16_t*, ptrdiff_t, const uint16_t*, ptrdiff_t, int, int, int16_t*, int);
void put_hevc_qpel_3_1_16_sse4(int16_t*, ptrdiff_t, const uint16_t*, ptrdiff_t, int, int, int16_t*, int);
void put_hevc_qpel_3_2_16_sse4(int16_t*, ptrdiff_t, const uint16_t*, ptrdiff_t, int, int, int16_t*, int);
void put_hevc_qpel_3_3_16_sse4(int16_t*, ptrdiff_t, const uint16_t*, ptrdiff_t, int, int, int16_t*, int);

#endif
