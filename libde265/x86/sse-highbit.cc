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

// SSE4.1 implementations of the high-bit-depth decoding kernels.  The scalar
// code uses int16_t prediction samples for bit depths up to 12, so the motion
// compensation and weighted prediction kernels below use int32_t accumulators
// internally but keep the same int16_t interface.  Results are bit-exact with
// the fallback implementations.

#include "x86/sse-highbit.h"

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#if HAVE_SSE4_1

#include "x86/sse-dct.h"
#include "fallback-dct.h"
#include "fallback-motion.h"
#include "fallback-intrapred.h"
#include "fallback-deblk.h"
#include "libde265/util.h"

#include <emmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>

#include <algorithm>
#include <string.h>

namespace {

inline __m128i load_u16x8(const uint16_t* p)
{
  return _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
}

inline void store_u16x8(uint16_t* p, __m128i v)
{
  _mm_storeu_si128(reinterpret_cast<__m128i*>(p), v);
}

inline __m128i load4_u16_i32(const uint16_t* p)
{
  return _mm_cvtepu16_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p)));
}

inline __m128i load4_i16_i32(const int16_t* p)
{
  return _mm_cvtepi16_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p)));
}

inline void store4_u16_i32(uint16_t* p, __m128i v)
{
  __m128i u = _mm_packus_epi32(v, v);
  _mm_storel_epi64(reinterpret_cast<__m128i*>(p), u);
}

inline __m128i sra_var_epi32(__m128i v, int shift)
{
  return _mm_sra_epi32(v, _mm_cvtsi32_si128(shift));
}

inline __m128i sll_var_epi16(__m128i v, int shift)
{
  return _mm_sll_epi16(v, _mm_cvtsi32_si128(shift));
}

inline __m128i clip_i32(__m128i v, int maxval)
{
  v = _mm_max_epi32(v, _mm_setzero_si128());
  return _mm_min_epi32(v, _mm_set1_epi32(maxval));
}

// store the low 'count' int16 values of v (count is 2,4,8)
inline void store_i16_partial(int16_t* dst, __m128i v, int count)
{
  switch (count) {
    case 2: {
      uint32_t t = static_cast<uint32_t>(_mm_cvtsi128_si32(v));
      memcpy(dst, &t, 4);
      break;
    }
    case 4:
      _mm_storel_epi64(reinterpret_cast<__m128i*>(dst), v);
      break;
    default:
      _mm_storeu_si128(reinterpret_cast<__m128i*>(dst), v);
      break;
  }
}

// store the low 'count' uint16 values of v (count is 2,4,8)
inline void store_u16_partial(uint16_t* dst, __m128i v, int count)
{
  store_i16_partial(reinterpret_cast<int16_t*>(dst), v, count);
}

const int16_t epel_filter_16[7][4] = {
  { -2, 58, 10, -2 },
  { -4, 54, 16, -2 },
  { -6, 46, 28, -4 },
  { -4, 36, 36, -4 },
  { -4, 28, 46, -6 },
  { -2, 16, 54, -4 },
  { -2, 10, 58, -2 },
};

const int16_t qpel_filter_16[4][8] = {
  { 0, 0, 0, 0, 0, 0, 0, 0 },
  { -1, 4, -10, 58, 17, -5, 1, 0 },
  { -1, 4, -11, 40, 40, -11, 4, -1 },
  { 1, -5, 17, 58, -10, 4, -1, 0 },
};

const int qpel_extra_before_16[4] = { 0, 3, 3, 2 };
const int qpel_extra_after_16[4]  = { 0, 3, 4, 4 };

// horizontal filter for 8 output samples.  'p' points at the first tap for the
// first output sample.  All filters have an even number of taps (4 or 8).
inline __m128i hfilter8(const uint16_t* p, const int16_t* filter, int ntaps, int shift)
{
  __m128i acc_lo = _mm_setzero_si128();
  __m128i acc_hi = _mm_setzero_si128();

  for (int i = 0; i < ntaps; i += 2) {
    __m128i a = load_u16x8(p + i);
    __m128i b = load_u16x8(p + i + 1);

    uint32_t pair = static_cast<uint16_t>(filter[i]) |
                    (static_cast<uint32_t>(static_cast<uint16_t>(filter[i+1])) << 16);
    __m128i c = _mm_set1_epi32(static_cast<int>(pair));

    acc_lo = _mm_add_epi32(acc_lo, _mm_madd_epi16(_mm_unpacklo_epi16(a, b), c));
    acc_hi = _mm_add_epi32(acc_hi, _mm_madd_epi16(_mm_unpackhi_epi16(a, b), c));
  }

  acc_lo = sra_var_epi32(acc_lo, shift);
  acc_hi = sra_var_epi32(acc_hi, shift);
  return _mm_packs_epi32(acc_lo, acc_hi);
}

// vertical filter for 8 output samples.
inline __m128i vfilter8(const int16_t* p, ptrdiff_t stride,
                        const int16_t* filter, int ntaps, int shift)
{
  __m128i acc_lo = _mm_setzero_si128();
  __m128i acc_hi = _mm_setzero_si128();

  for (int i = 0; i < ntaps; i += 2) {
    __m128i a = load_u16x8(reinterpret_cast<const uint16_t*>(p + i*stride));
    __m128i b = load_u16x8(reinterpret_cast<const uint16_t*>(p + (i+1)*stride));

    uint32_t pair = static_cast<uint16_t>(filter[i]) |
                    (static_cast<uint32_t>(static_cast<uint16_t>(filter[i+1])) << 16);
    __m128i c = _mm_set1_epi32(static_cast<int>(pair));

    acc_lo = _mm_add_epi32(acc_lo, _mm_madd_epi16(_mm_unpacklo_epi16(a, b), c));
    acc_hi = _mm_add_epi32(acc_hi, _mm_madd_epi16(_mm_unpackhi_epi16(a, b), c));
  }

  acc_lo = sra_var_epi32(acc_lo, shift);
  acc_hi = sra_var_epi32(acc_hi, shift);
  return _mm_packs_epi32(acc_lo, acc_hi);
}

// Filter a complete row into dst, handling widths that are not a multiple of 8.
inline void filter_row_h(int16_t* dst, const uint16_t* src, int width,
                         const int16_t* filter, int ntaps, int shift)
{
  int x = 0;
  for (; x + 8 <= width; x += 8) {
    store_u16x8(reinterpret_cast<uint16_t*>(dst + x),
                hfilter8(src + x, filter, ntaps, shift));
  }
  if (x + 4 <= width) {
    store_i16_partial(dst + x, hfilter8(src + x, filter, ntaps, shift), 4);
    x += 4;
  }

  // The remaining samples (2 at most in valid HEVC block sizes) use the
  // scalar formula so narrow sub-blocks are still handled correctly.
  for (; x < width; x++) {
    int sum = 0;
    for (int k = 0; k < ntaps; k++) {
      sum += filter[k] * src[x+k];
    }
    dst[x] = static_cast<int16_t>(sum >> shift);
  }
}

inline void copy_shift_samples(uint16_t* dst, ptrdiff_t dststride,
                               const uint16_t* src, ptrdiff_t srcstride,
                               int width, int height, int shift)
{
  for (int y = 0; y < height; y++) {
    const uint16_t* s = src + y*srcstride;
    uint16_t* d = dst + y*dststride;
    int x = 0;
    for (; x + 8 <= width; x += 8) {
      store_u16x8(d + x, sll_var_epi16(load_u16x8(s + x), shift));
    }
    for (; x < width; x++) {
      d[x] = static_cast<uint16_t>(s[x] << shift);
    }
  }
}

} // namespace


// --- weighted prediction ---------------------------------------------------

void put_unweighted_pred_16_sse4(uint16_t *dst, ptrdiff_t dststride,
                                 const int16_t *src, ptrdiff_t srcstride,
                                 int width, int height, int bit_depth)
{
  const int shift = std::max(2, 14 - bit_depth);
  const int offset = 1 << (shift - 1);
  const int maxval = (1 << bit_depth) - 1;
  const __m128i voff = _mm_set1_epi32(offset);

  for (int y = 0; y < height; y++) {
    const int16_t* in = src + y*srcstride;
    uint16_t* out = dst + y*dststride;
    int x = 0;
    for (; x + 8 <= width; x += 8) {
      __m128i v0 = _mm_add_epi32(load4_i16_i32(in + x), voff);
      __m128i v1 = _mm_add_epi32(load4_i16_i32(in + x + 4), voff);
      v0 = clip_i32(sra_var_epi32(v0, shift), maxval);
      v1 = clip_i32(sra_var_epi32(v1, shift), maxval);
      store_u16x8(out + x, _mm_packus_epi32(v0, v1));
    }
    if (x + 4 <= width) {
      __m128i v = _mm_add_epi32(load4_i16_i32(in + x), voff);
      v = clip_i32(sra_var_epi32(v, shift), maxval);
      store4_u16_i32(out + x, v);
      x += 4;
    }
    for (; x < width; x++) {
      out[x] = static_cast<uint16_t>(Clip_BitDepth((in[x] + offset) >> shift, bit_depth));
    }
  }
}


void put_weighted_pred_avg_16_sse4(uint16_t *dst, ptrdiff_t dststride,
                                   const int16_t *src1, const int16_t *src2,
                                   ptrdiff_t srcstride, int width, int height,
                                   int bit_depth)
{
  const int shift = std::max(3, 15 - bit_depth);
  const int offset = 1 << (shift - 1);
  const int maxval = (1 << bit_depth) - 1;
  const __m128i voff = _mm_set1_epi32(offset);

  for (int y = 0; y < height; y++) {
    const int16_t* in1 = src1 + y*srcstride;
    const int16_t* in2 = src2 + y*srcstride;
    uint16_t* out = dst + y*dststride;
    int x = 0;
    for (; x + 8 <= width; x += 8) {
      __m128i a0 = load4_i16_i32(in1 + x);
      __m128i a1 = load4_i16_i32(in1 + x + 4);
      __m128i b0 = load4_i16_i32(in2 + x);
      __m128i b1 = load4_i16_i32(in2 + x + 4);
      __m128i v0 = _mm_add_epi32(_mm_add_epi32(a0, b0), voff);
      __m128i v1 = _mm_add_epi32(_mm_add_epi32(a1, b1), voff);
      v0 = clip_i32(sra_var_epi32(v0, shift), maxval);
      v1 = clip_i32(sra_var_epi32(v1, shift), maxval);
      store_u16x8(out + x, _mm_packus_epi32(v0, v1));
    }
    if (x + 4 <= width) {
      __m128i a = load4_i16_i32(in1 + x);
      __m128i b = load4_i16_i32(in2 + x);
      __m128i v = _mm_add_epi32(_mm_add_epi32(a, b), voff);
      v = clip_i32(sra_var_epi32(v, shift), maxval);
      store4_u16_i32(out + x, v);
      x += 4;
    }
    for (; x < width; x++) {
      out[x] = static_cast<uint16_t>(Clip_BitDepth((in1[x] + in2[x] + offset) >> shift, bit_depth));
    }
  }
}


void put_weighted_pred_16_sse4(uint16_t *dst, ptrdiff_t dststride,
                               const int16_t *src, ptrdiff_t srcstride,
                               int width, int height,
                               int w, int o, int log2WD, int bit_depth)
{
  const int rnd = 1 << (log2WD - 1);
  const int maxval = (1 << bit_depth) - 1;
  const __m128i vw = _mm_set1_epi32(w);
  const __m128i vrnd = _mm_set1_epi32(rnd);
  const __m128i vo = _mm_set1_epi32(o);

  for (int y = 0; y < height; y++) {
    const int16_t* in = src + y*srcstride;
    uint16_t* out = dst + y*dststride;
    int x = 0;
    for (; x + 8 <= width; x += 8) {
      __m128i a0 = load4_i16_i32(in + x);
      __m128i a1 = load4_i16_i32(in + x + 4);
      __m128i v0 = _mm_add_epi32(sra_var_epi32(
                      _mm_add_epi32(_mm_mullo_epi32(a0, vw), vrnd), log2WD), vo);
      __m128i v1 = _mm_add_epi32(sra_var_epi32(
                      _mm_add_epi32(_mm_mullo_epi32(a1, vw), vrnd), log2WD), vo);
      v0 = clip_i32(v0, maxval);
      v1 = clip_i32(v1, maxval);
      store_u16x8(out + x, _mm_packus_epi32(v0, v1));
    }
    if (x + 4 <= width) {
      __m128i a = load4_i16_i32(in + x);
      __m128i v = _mm_add_epi32(sra_var_epi32(
                      _mm_add_epi32(_mm_mullo_epi32(a, vw), vrnd), log2WD), vo);
      v = clip_i32(v, maxval);
      store4_u16_i32(out + x, v);
      x += 4;
    }
    for (; x < width; x++) {
      out[x] = static_cast<uint16_t>(Clip_BitDepth(((in[x]*w + rnd) >> log2WD) + o, bit_depth));
    }
  }
}


void put_weighted_bipred_16_sse4(uint16_t *dst, ptrdiff_t dststride,
                                 const int16_t *src1, const int16_t *src2,
                                 ptrdiff_t srcstride, int width, int height,
                                 int w1, int o1, int w2, int o2, int log2WD,
                                 int bit_depth)
{
  const int rnd = static_cast<int>(static_cast<unsigned int>(o1 + o2 + 1) << log2WD);
  const int maxval = (1 << bit_depth) - 1;
  const __m128i vw1 = _mm_set1_epi32(w1);
  const __m128i vw2 = _mm_set1_epi32(w2);
  const __m128i vrnd = _mm_set1_epi32(rnd);

  for (int y = 0; y < height; y++) {
    const int16_t* in1 = src1 + y*srcstride;
    const int16_t* in2 = src2 + y*srcstride;
    uint16_t* out = dst + y*dststride;
    int x = 0;
    for (; x + 8 <= width; x += 8) {
      __m128i a0 = load4_i16_i32(in1 + x);
      __m128i a1 = load4_i16_i32(in1 + x + 4);
      __m128i b0 = load4_i16_i32(in2 + x);
      __m128i b1 = load4_i16_i32(in2 + x + 4);
      __m128i v0 = _mm_add_epi32(_mm_add_epi32(_mm_mullo_epi32(a0, vw1),
                                               _mm_mullo_epi32(b0, vw2)), vrnd);
      __m128i v1 = _mm_add_epi32(_mm_add_epi32(_mm_mullo_epi32(a1, vw1),
                                               _mm_mullo_epi32(b1, vw2)), vrnd);
      v0 = clip_i32(sra_var_epi32(v0, log2WD + 1), maxval);
      v1 = clip_i32(sra_var_epi32(v1, log2WD + 1), maxval);
      store_u16x8(out + x, _mm_packus_epi32(v0, v1));
    }
    if (x + 4 <= width) {
      __m128i a = load4_i16_i32(in1 + x);
      __m128i b = load4_i16_i32(in2 + x);
      __m128i v = _mm_add_epi32(_mm_add_epi32(_mm_mullo_epi32(a, vw1),
                                              _mm_mullo_epi32(b, vw2)), vrnd);
      v = clip_i32(sra_var_epi32(v, log2WD + 1), maxval);
      store4_u16_i32(out + x, v);
      x += 4;
    }
    for (; x < width; x++) {
      out[x] = static_cast<uint16_t>(Clip_BitDepth(
          (in1[x]*w1 + in2[x]*w2 + rnd) >> (log2WD + 1), bit_depth));
    }
  }
}


// --- motion compensation ---------------------------------------------------

void put_hevc_epel_16_sse4(int16_t *dst, ptrdiff_t dststride,
                           const uint16_t *src, ptrdiff_t srcstride,
                           int width, int height,
                           int mx, int my, int16_t *mcbuffer, int bit_depth)
{
  (void)mx; (void)my; (void)mcbuffer;
  const int shift3 = std::max(2, 14 - bit_depth);
  copy_shift_samples(reinterpret_cast<uint16_t*>(dst), dststride,
                     src, srcstride, width, height, shift3);
}


void put_hevc_epel_h_16_sse4(int16_t *dst, ptrdiff_t dststride,
                             const uint16_t *src, ptrdiff_t srcstride,
                             int width, int height,
                             int mx, int my, int16_t *mcbuffer, int bit_depth)
{
  (void)my; (void)mcbuffer;
  const int shift1 = std::min(4, bit_depth - 8);
  const int16_t* filter = epel_filter_16[mx - 1];

  for (int y = 0; y < height; y++) {
    filter_row_h(dst + y*dststride, src + y*srcstride - 1, width,
                 filter, 4, shift1);
  }
}


void put_hevc_epel_v_16_sse4(int16_t *dst, ptrdiff_t dststride,
                             const uint16_t *src, ptrdiff_t srcstride,
                             int width, int height,
                             int mx, int my, int16_t *mcbuffer, int bit_depth)
{
  (void)mx; (void)mcbuffer;
  const int shift = std::min(4, bit_depth - 8);
  const int16_t* filter = epel_filter_16[my - 1];

  for (int y = 0; y < height; y++) {
    const uint16_t* in = src + (y - 1)*srcstride;
    int16_t* out = dst + y*dststride;
    int x = 0;
    for (; x + 8 <= width; x += 8) {
      store_u16x8(reinterpret_cast<uint16_t*>(out + x),
                  vfilter8(reinterpret_cast<const int16_t*>(in + x), srcstride,
                           filter, 4, shift));
    }
    for (; x < width; x++) {
      int sum = 0;
      for (int k = 0; k < 4; k++) {
        sum += filter[k] * in[x + k*srcstride];
      }
      out[x] = static_cast<int16_t>(sum >> shift);
    }
  }
}


void put_hevc_epel_hv_16_sse4(int16_t *dst, ptrdiff_t dststride,
                              const uint16_t *src, ptrdiff_t srcstride,
                              int width, int height,
                              int mx, int my, int16_t *mcbuffer, int bit_depth)
{
  const int shift1 = std::min(4, bit_depth - 8);
  const int shift2 = 6;
  const int16_t* filter_h = epel_filter_16[mx - 1];
  const int16_t* filter_v = epel_filter_16[my - 1];

  const int tmp_stride = (width + 7) & ~7;
  const int nRows = height + 3;

  for (int yy = 0; yy < nRows; yy++) {
    filter_row_h(mcbuffer + yy*tmp_stride, src + (yy - 1)*srcstride - 1,
                 width, filter_h, 4, shift1);
    for (int x = width; x < tmp_stride; x++) {
      mcbuffer[yy*tmp_stride + x] = 0;
    }
  }

  for (int y = 0; y < height; y++) {
    int16_t* out = dst + y*dststride;
    int x = 0;
    for (; x + 8 <= width; x += 8) {
      store_u16x8(reinterpret_cast<uint16_t*>(out + x),
                  vfilter8(mcbuffer + y*tmp_stride + x, tmp_stride,
                           filter_v, 4, shift2));
    }
    for (; x < width; x++) {
      int sum = 0;
      for (int k = 0; k < 4; k++) {
        sum += filter_v[k] * mcbuffer[(y+k)*tmp_stride + x];
      }
      out[x] = static_cast<int16_t>(sum >> shift2);
    }
  }
}


void put_hevc_qpel_16_sse4(int16_t *dst, ptrdiff_t dststride,
                           const uint16_t *src, ptrdiff_t srcstride,
                           int width, int height, int16_t *mcbuffer,
                           int xFrac, int yFrac, int bit_depth)
{
  if (xFrac == 0 && yFrac == 0) {
    copy_shift_samples(reinterpret_cast<uint16_t*>(dst), dststride,
                       src, srcstride, width, height,
                       std::max(2, 14 - bit_depth));
    return;
  }

  const int shift1 = std::min(4, bit_depth - 8);
  const int shift2 = 6;

  const int extra_left = qpel_extra_before_16[xFrac];
  const int extra_top = qpel_extra_before_16[yFrac];
  const int extra_bottom = qpel_extra_after_16[yFrac];

  if (xFrac != 0 && yFrac == 0) {
    const int16_t* filter = qpel_filter_16[xFrac];
    for (int y = 0; y < height; y++) {
      filter_row_h(dst + y*dststride, src + y*srcstride - extra_left, width,
                   filter, 8, shift1);
    }
    return;
  }

  if (xFrac == 0 && yFrac != 0) {
    const int16_t* filter = qpel_filter_16[yFrac];
    for (int y = 0; y < height; y++) {
      const uint16_t* in = src + (y - extra_top)*srcstride;
      int16_t* out = dst + y*dststride;
      int x = 0;
      for (; x + 8 <= width; x += 8) {
        store_u16x8(reinterpret_cast<uint16_t*>(out + x),
                    vfilter8(reinterpret_cast<const int16_t*>(in + x), srcstride,
                             filter, 8, shift1));
      }
      for (; x < width; x++) {
        int sum = 0;
        for (int k = 0; k < 8; k++) {
          sum += filter[k] * in[x + k*srcstride];
        }
        out[x] = static_cast<int16_t>(sum >> shift1);
      }
    }
    return;
  }

  // Both directions fractional: horizontal pass into the scratch buffer, then
  // the vertical pass.  The buffer is used row-major; the fallback uses a
  // column-major layout but the layout is private to the kernel.
  const int tmp_stride = (width + 7) & ~7;
  const int nRows = height + extra_top + extra_bottom;

  for (int yy = 0; yy < nRows; yy++) {
    filter_row_h(mcbuffer + yy*tmp_stride,
                 src + (yy - extra_top)*srcstride - extra_left,
                 width, qpel_filter_16[xFrac], 8, shift1);
    for (int x = width; x < tmp_stride; x++) {
      mcbuffer[yy*tmp_stride + x] = 0;
    }
  }

  for (int y = 0; y < height; y++) {
    int16_t* out = dst + y*dststride;
    int x = 0;
    for (; x + 8 <= width; x += 8) {
      store_u16x8(reinterpret_cast<uint16_t*>(out + x),
                  vfilter8(mcbuffer + y*tmp_stride + x, tmp_stride,
                           qpel_filter_16[yFrac], 8, shift2));
    }
    for (; x < width; x++) {
      int sum = 0;
      for (int k = 0; k < 8; k++) {
        sum += qpel_filter_16[yFrac][k] * mcbuffer[(y+k)*tmp_stride + x];
      }
      out[x] = static_cast<int16_t>(sum >> shift2);
    }
  }
}


#define QPEL16_WRAPPER(x,y)                                                   \
  void put_hevc_qpel_##x##_##y##_16_sse4(int16_t *dst, ptrdiff_t dststride,   \
                                         const uint16_t *src,                 \
                                         ptrdiff_t srcstride,                 \
                                         int width, int height,               \
                                         int16_t *mcbuffer, int bit_depth) {  \
    put_hevc_qpel_16_sse4(dst, dststride, src, srcstride, width, height,      \
                          mcbuffer, x, y, bit_depth);                         \
  }

QPEL16_WRAPPER(0,0) QPEL16_WRAPPER(0,1) QPEL16_WRAPPER(0,2) QPEL16_WRAPPER(0,3)
QPEL16_WRAPPER(1,0) QPEL16_WRAPPER(1,1) QPEL16_WRAPPER(1,2) QPEL16_WRAPPER(1,3)
QPEL16_WRAPPER(2,0) QPEL16_WRAPPER(2,1) QPEL16_WRAPPER(2,2) QPEL16_WRAPPER(2,3)
QPEL16_WRAPPER(3,0) QPEL16_WRAPPER(3,1) QPEL16_WRAPPER(3,2) QPEL16_WRAPPER(3,3)

#undef QPEL16_WRAPPER


// --- intra prediction ------------------------------------------------------

namespace {

const int kMaxIntraBlk = 64;

inline int intra_shift(int nT)
{
  return (nT == 4) ? 3 : (nT == 8) ? 4 : (nT == 16) ? 5 : (nT == 32) ? 6 : 7;
}

} // namespace


void intra_pred_dc_16_sse4(uint16_t* dst, ptrdiff_t stride, int nT, int cIdx,
                           const uint16_t* border)
{
  const int shift = intra_shift(nT);

  int dcVal = 0;
  for (int i = 0; i < nT; i++) {
    dcVal += border[i+1];
    dcVal += border[-i-1];
  }
  dcVal = (dcVal + nT) >> shift;

  const __m128i v = _mm_set1_epi16(static_cast<short>(dcVal));
  for (int y = 0; y < nT; y++) {
    uint16_t* d = dst + y*stride;
    int x = 0;
    for (; x + 8 <= nT; x += 8) {
      store_u16x8(d + x, v);
    }
    for (; x < nT; x++) {
      d[x] = static_cast<uint16_t>(dcVal);
    }
  }

  if (cIdx == 0 && nT < 32) {
    dst[0] = static_cast<uint16_t>((border[-1] + 2*dcVal + border[1] + 2) >> 2);
    for (int x = 1; x < nT; x++) {
      dst[x] = static_cast<uint16_t>((border[x+1] + 3*dcVal + 2) >> 2);
    }
    for (int y = 1; y < nT; y++) {
      dst[y*stride] = static_cast<uint16_t>((border[-y-1] + 3*dcVal + 2) >> 2);
    }
  }
}


void intra_pred_planar_16_sse4(uint16_t* dst, ptrdiff_t stride, int nT, int cIdx,
                               const uint16_t* border)
{
  (void)cIdx;
  const int shift = intra_shift(nT);
  const int TR = border[1+nT];
  const int BL = border[-1-nT];

  for (int y = 0; y < nT; y++) {
    const int left_y = border[-1-y];
    const int Cy = (y+1)*BL + nT;
    const __m128i vL = _mm_set1_epi32(left_y);
    const __m128i vTR = _mm_set1_epi32(TR);
    const __m128i vNT1Y = _mm_set1_epi32(nT-1-y);
    const __m128i vC = _mm_set1_epi32(Cy);
    uint16_t* d = dst + y*stride;

    int x = 0;
    for (; x + 8 <= nT; x += 8) {
      __m128i lo = load4_u16_i32(border + 1 + x);
      __m128i hi = load4_u16_i32(border + 1 + x + 4);
      __m128i idx0 = _mm_setr_epi32(x+0, x+1, x+2, x+3);
      __m128i idx1 = _mm_setr_epi32(x+4, x+5, x+6, x+7);
      __m128i vA0 = _mm_sub_epi32(_mm_set1_epi32(nT-1), idx0);
      __m128i vA1 = _mm_sub_epi32(_mm_set1_epi32(nT-1), idx1);
      __m128i vB0 = _mm_add_epi32(idx0, _mm_set1_epi32(1));
      __m128i vB1 = _mm_add_epi32(idx1, _mm_set1_epi32(1));
      __m128i acc0 = _mm_add_epi32(_mm_mullo_epi32(vA0, vL), vC);
      __m128i acc1 = _mm_add_epi32(_mm_mullo_epi32(vA1, vL), vC);
      acc0 = _mm_add_epi32(acc0, _mm_mullo_epi32(vB0, vTR));
      acc1 = _mm_add_epi32(acc1, _mm_mullo_epi32(vB1, vTR));
      acc0 = _mm_add_epi32(acc0, _mm_mullo_epi32(lo, vNT1Y));
      acc1 = _mm_add_epi32(acc1, _mm_mullo_epi32(hi, vNT1Y));
      acc0 = sra_var_epi32(acc0, shift);
      acc1 = sra_var_epi32(acc1, shift);
      store_u16x8(d + x, _mm_packus_epi32(acc0, acc1));
    }
    for (; x < nT; x++) {
      int v = (nT-1-x)*left_y + (x+1)*TR + (nT-1-y)*border[1+x] +
              (y+1)*BL + nT;
      d[x] = static_cast<uint16_t>(v >> shift);
    }
  }
}


void intra_pred_angular_16_sse4(uint16_t* dst, ptrdiff_t stride, int bit_depth,
                                int disableBoundaryFilter, int xB0, int yB0,
                                int mode, int nT, int cIdx,
                                const uint16_t* border)
{
  (void)xB0; (void)yB0;
  extern const int intraPredAngle_table[1+34];
  extern const int invAngle_table[25-10];

  const int intraPredAngle = intraPredAngle_table[mode];
  uint16_t ref_mem[4*kMaxIntraBlk+1];
  uint16_t* ref = &ref_mem[2*kMaxIntraBlk];

  if (mode >= 18) {
    for (int x = 0; x <= nT; x++) {
      ref[x] = border[x];
    }

    if (intraPredAngle < 0) {
      const int invAngle = invAngle_table[mode-11];
      if (((nT*intraPredAngle) >> 5) < -1) {
        for (int x = (nT*intraPredAngle)>>5; x <= -1; x++) {
          ref[x] = border[0 - ((x*invAngle + 128) >> 8)];
        }
      }
    }
    else {
      for (int x = nT+1; x <= 2*nT; x++) {
        ref[x] = border[x];
      }
    }

    for (int y = 0; y < nT; y++) {
      const int iIdx = ((y+1)*intraPredAngle) >> 5;
      const int iFact = ((y+1)*intraPredAngle) & 31;
      const uint16_t* src = ref + iIdx + 1;
      uint16_t* d = dst + y*stride;

      if (iFact == 0) {
        for (int x = 0; x < nT; x++) {
          d[x] = src[x];
        }
      }
      else {
        const int w0 = 32 - iFact;
        const int w1 = iFact;
        int x = 0;
        for (; x + 8 <= nT; x += 8) {
          __m128i a = load4_u16_i32(src + x);
          __m128i b = load4_u16_i32(src + x + 1);
          __m128i c = load4_u16_i32(src + x + 4);
          __m128i e = load4_u16_i32(src + x + 5);
          __m128i v0 = _mm_add_epi32(_mm_add_epi32(_mm_mullo_epi32(a, _mm_set1_epi32(w0)),
                                                   _mm_mullo_epi32(b, _mm_set1_epi32(w1))),
                                     _mm_set1_epi32(16));
          __m128i v1 = _mm_add_epi32(_mm_add_epi32(_mm_mullo_epi32(c, _mm_set1_epi32(w0)),
                                                   _mm_mullo_epi32(e, _mm_set1_epi32(w1))),
                                     _mm_set1_epi32(16));
          v0 = sra_var_epi32(v0, 5);
          v1 = sra_var_epi32(v1, 5);
          store_u16x8(d + x, _mm_packus_epi32(v0, v1));
        }
        for (; x < nT; x++) {
          d[x] = static_cast<uint16_t>((w0*src[x] + w1*src[x+1] + 16) >> 5);
        }
      }
    }

    if (mode == 26 && cIdx == 0 && nT < 32 && !disableBoundaryFilter) {
      for (int y = 0; y < nT; y++) {
        dst[y*stride] = static_cast<uint16_t>(
            Clip_BitDepth(border[1] + ((border[-1-y] - border[0]) >> 1), bit_depth));
      }
    }
  }
  else {
    // Modes 2..17 project along the left border.  As in the 8-bit SSE
    // implementation, keep the scalar loop here; the projection is
    // transposed and does not map to contiguous loads/stores.
    for (int x = 0; x <= nT; x++) {
      ref[x] = border[-x];
    }

    if (intraPredAngle < 0) {
      const int invAngle = invAngle_table[mode-11];
      if (((nT*intraPredAngle) >> 5) < -1) {
        for (int x = (nT*intraPredAngle)>>5; x <= -1; x++) {
          ref[x] = border[(x*invAngle + 128) >> 8];
        }
      }
    }
    else {
      for (int x = nT+1; x <= 2*nT; x++) {
        ref[x] = border[-x];
      }
    }

    for (int y = 0; y < nT; y++) {
      for (int x = 0; x < nT; x++) {
        const int iIdx = ((x+1)*intraPredAngle) >> 5;
        const int iFact = ((x+1)*intraPredAngle) & 31;
        if (iFact != 0) {
          dst[x + y*stride] = static_cast<uint16_t>(
              ((32-iFact)*ref[y+iIdx+1] + iFact*ref[y+iIdx+2] + 16) >> 5);
        }
        else {
          dst[x + y*stride] = ref[y+iIdx+1];
        }
      }
    }

    if (mode == 10 && cIdx == 0 && nT < 32 && !disableBoundaryFilter) {
      for (int x = 0; x < nT; x++) {
        dst[x] = static_cast<uint16_t>(
            Clip_BitDepth(border[-1] + ((border[1+x] - border[0]) >> 1), bit_depth));
      }
    }
  }
}


// --- deblocking ------------------------------------------------------------

namespace {

inline __m128i clip_pixel_i32(__m128i v, int maxval)
{
  return clip_i32(v, maxval);
}

inline __m128i x2_32(__m128i a) { return _mm_add_epi32(a, a); }
inline __m128i add3_32(__m128i a, __m128i b, __m128i c)
{
  return _mm_add_epi32(_mm_add_epi32(a, b), c);
}

inline __m128i clip3_32(__m128i lo, __m128i hi, __m128i v)
{
  return _mm_min_epi32(_mm_max_epi32(v, lo), hi);
}

// Load 4 lines x 8 samples and transpose into eight vectors of 4 int32.
inline void load_vert16(const uint16_t* ptr, ptrdiff_t stride, __m128i s[8])
{
  __m128i r0 = load_u16x8(ptr - 4);
  __m128i r1 = load_u16x8(ptr - 4 + stride);
  __m128i r2 = load_u16x8(ptr - 4 + 2*stride);
  __m128i r3 = load_u16x8(ptr - 4 + 3*stride);

  __m128i t0 = _mm_unpacklo_epi16(r0, r1);
  __m128i t1 = _mm_unpacklo_epi16(r2, r3);
  __m128i u0 = _mm_unpacklo_epi32(t0, t1);  // samples 0,1
  __m128i u1 = _mm_unpackhi_epi32(t0, t1);  // samples 2,3

  t0 = _mm_unpackhi_epi16(r0, r1);
  t1 = _mm_unpackhi_epi16(r2, r3);
  __m128i u2 = _mm_unpacklo_epi32(t0, t1);  // samples 4,5
  __m128i u3 = _mm_unpackhi_epi32(t0, t1);  // samples 6,7

  __m128i u[4] = { u0, u1, u2, u3 };
  for (int j = 0; j < 8; j++) {
    __m128i x = u[j/2];
    if (j & 1) {
      x = _mm_srli_si128(x, 8);
    }
    s[j] = _mm_cvtepu16_epi32(x);
  }
}

inline void store_vert16(uint16_t* ptr, ptrdiff_t stride, const __m128i s[8])
{
  alignas(16) uint16_t vals[8][4];
  for (int j = 0; j < 8; j++) {
    __m128i v = _mm_packus_epi32(s[j], s[j]);
    _mm_storel_epi64(reinterpret_cast<__m128i*>(vals[j]), v);
  }

  for (int k = 0; k < 4; k++) {
    uint16_t* row = ptr - 4 + k*stride;
    for (int j = 0; j < 8; j++) {
      row[j] = vals[j][k];
    }
  }
}

inline void load_horiz16(const uint16_t* ptr, ptrdiff_t stride, __m128i s[8])
{
  for (int j = 0; j < 8; j++) {
    s[j] = load4_u16_i32(ptr + (j-4)*stride);
  }
}

inline void store_horiz16(uint16_t* ptr, ptrdiff_t stride, const __m128i s[8])
{
  for (int j = 0; j < 8; j++) {
    store4_u16_i32(ptr + (j-4)*stride, s[j]);
  }
}

} // namespace


void deblock_luma_16_sse4(uint16_t* ptr, ptrdiff_t stride, int vertical,
                          int dE, int dEp, int dEq, int tc,
                          int filterP, int filterQ, int bitDepth)
{
  __m128i s[8];
  if (vertical) load_vert16(ptr, stride, s); else load_horiz16(ptr, stride, s);

  const __m128i p3 = s[0], p2 = s[1], p1 = s[2], p0 = s[3];
  const __m128i q0 = s[4], q1 = s[5], q2 = s[6], q3 = s[7];
  const int maxval = (1 << bitDepth) - 1;

  if (dE == 2) {
    const __m128i v2tc = _mm_set1_epi32(2*tc);
    const __m128i c4 = _mm_set1_epi32(4);
    const __m128i c2 = _mm_set1_epi32(2);

    __m128i pn0 = _mm_srai_epi32(_mm_add_epi32(add3_32(p2, x2_32(p1), x2_32(p0)),
                                               add3_32(x2_32(q0), q1, c4)), 3);
    pn0 = clip3_32(_mm_sub_epi32(p0, v2tc), _mm_add_epi32(p0, v2tc), pn0);
    __m128i pn1 = _mm_srai_epi32(_mm_add_epi32(add3_32(p2, p1, p0),
                                               _mm_add_epi32(q0, c2)), 2);
    pn1 = clip3_32(_mm_sub_epi32(p1, v2tc), _mm_add_epi32(p1, v2tc), pn1);
    __m128i pn2 = _mm_srai_epi32(_mm_add_epi32(add3_32(x2_32(p3), _mm_add_epi32(x2_32(p2), p2), p1),
                                               add3_32(p0, q0, c4)), 3);
    pn2 = clip3_32(_mm_sub_epi32(p2, v2tc), _mm_add_epi32(p2, v2tc), pn2);

    __m128i qn0 = _mm_srai_epi32(_mm_add_epi32(add3_32(p1, x2_32(p0), x2_32(q0)),
                                               add3_32(x2_32(q1), q2, c4)), 3);
    qn0 = clip3_32(_mm_sub_epi32(q0, v2tc), _mm_add_epi32(q0, v2tc), qn0);
    __m128i qn1 = _mm_srai_epi32(_mm_add_epi32(add3_32(p0, q0, q1),
                                               _mm_add_epi32(q2, c2)), 2);
    qn1 = clip3_32(_mm_sub_epi32(q1, v2tc), _mm_add_epi32(q1, v2tc), qn1);
    __m128i qn2 = _mm_srai_epi32(_mm_add_epi32(add3_32(p0, q0, q1),
                                               add3_32(_mm_add_epi32(x2_32(q2), q2), x2_32(q3), c4)), 3);
    qn2 = clip3_32(_mm_sub_epi32(q2, v2tc), _mm_add_epi32(q2, v2tc), qn2);

    if (filterP) { s[3] = pn0; s[2] = pn1; s[1] = pn2; }
    if (filterQ) { s[4] = qn0; s[5] = qn1; s[6] = qn2; }
  }
  else {
    const __m128i vtc = _mm_set1_epi32(tc);
    __m128i delta0 = _mm_srai_epi32(
        _mm_add_epi32(_mm_sub_epi32(_mm_mullo_epi32(_mm_set1_epi32(9), _mm_sub_epi32(q0, p0)),
                                    _mm_mullo_epi32(_mm_set1_epi32(3), _mm_sub_epi32(q1, p1))),
                      _mm_set1_epi32(8)), 4);
    __m128i mask = _mm_cmpgt_epi32(_mm_set1_epi32(tc*10), _mm_abs_epi32(delta0));
    __m128i delta = clip3_32(_mm_set1_epi32(-tc), vtc, delta0);

    if (filterP) {
      s[3] = _mm_blendv_epi8(p0, clip_pixel_i32(_mm_add_epi32(p0, delta), maxval), mask);
    }
    if (filterQ) {
      s[4] = _mm_blendv_epi8(q0, clip_pixel_i32(_mm_sub_epi32(q0, delta), maxval), mask);
    }
    if (dEp && filterP) {
      const __m128i htc = _mm_set1_epi32(tc >> 1);
      __m128i dp = _mm_srai_epi32(
          _mm_add_epi32(_mm_sub_epi32(_mm_srai_epi32(
              _mm_add_epi32(_mm_add_epi32(p2, p0), _mm_set1_epi32(1)), 1), p1), delta), 1);
      dp = clip3_32(_mm_sub_epi32(_mm_setzero_si128(), htc), htc, dp);
      s[2] = _mm_blendv_epi8(p1, clip_pixel_i32(_mm_add_epi32(p1, dp), maxval), mask);
    }
    if (dEq && filterQ) {
      const __m128i htc = _mm_set1_epi32(tc >> 1);
      __m128i dq = _mm_srai_epi32(
          _mm_sub_epi32(_mm_sub_epi32(_mm_srai_epi32(
              _mm_add_epi32(_mm_add_epi32(q2, q0), _mm_set1_epi32(1)), 1), q1), delta), 1);
      dq = clip3_32(_mm_sub_epi32(_mm_setzero_si128(), htc), htc, dq);
      s[5] = _mm_blendv_epi8(q1, clip_pixel_i32(_mm_add_epi32(q1, dq), maxval), mask);
    }
  }

  if (vertical) store_vert16(ptr, stride, s); else store_horiz16(ptr, stride, s);
}


// --- inverse transform wrappers -------------------------------------------

void transform_4x4_dst_add_16_sse4(uint16_t *dst, const int16_t *coeffs,
                                   ptrdiff_t stride, int bit_depth)
{
  if (bit_depth == 10) {
    ff_hevc_transform_4x4_luma_add_10_sse4(reinterpret_cast<uint8_t*>(dst),
                                           coeffs, stride*2);
  }
  else {
    transform_4x4_luma_add_16_fallback(dst, coeffs, stride, bit_depth);
  }
}

void transform_4x4_add_16_sse4(uint16_t *dst, const int16_t *coeffs,
                               ptrdiff_t stride, int bit_depth)
{
  if (bit_depth == 10) {
    alignas(16) int16_t tmp[4*4];
    memcpy(tmp, coeffs, sizeof(tmp));
    ff_hevc_transform_4x4_add_10_sse4(reinterpret_cast<uint8_t*>(dst),
                                      tmp, stride*2);
  }
  else {
    transform_4x4_add_16_fallback(dst, coeffs, stride, bit_depth);
  }
}

void transform_8x8_add_16_sse4(uint16_t *dst, const int16_t *coeffs,
                               ptrdiff_t stride, int bit_depth)
{
  if (bit_depth == 10) {
    alignas(16) int16_t tmp[8*8];
    memcpy(tmp, coeffs, sizeof(tmp));
    ff_hevc_transform_8x8_add_10_sse4(reinterpret_cast<uint8_t*>(dst),
                                      tmp, stride*2);
  }
  else {
    transform_8x8_add_16_fallback(dst, coeffs, stride, bit_depth);
  }
}

void transform_16x16_add_16_sse4(uint16_t *dst, const int16_t *coeffs,
                                 ptrdiff_t stride, int bit_depth)
{
  if (bit_depth == 10) {
    alignas(16) int16_t tmp[16*16];
    memcpy(tmp, coeffs, sizeof(tmp));
    ff_hevc_transform_16x16_add_10_sse4(reinterpret_cast<uint8_t*>(dst),
                                        tmp, stride*2);
  }
  else {
    transform_16x16_add_16_fallback(dst, coeffs, stride, bit_depth);
  }
}

void transform_32x32_add_16_sse4(uint16_t *dst, const int16_t *coeffs,
                                 ptrdiff_t stride, int bit_depth)
{
  if (bit_depth == 10) {
    alignas(16) int16_t tmp[32*32];
    memcpy(tmp, coeffs, sizeof(tmp));
    ff_hevc_transform_32x32_add_10_sse4(reinterpret_cast<uint8_t*>(dst),
                                        tmp, stride*2);
  }
  else {
    transform_32x32_add_16_fallback(dst, coeffs, stride, bit_depth);
  }
}

#endif // HAVE_SSE4_1
