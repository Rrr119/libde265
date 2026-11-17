/*
 * H.265 video codec.
 * Copyright (c) 2026 Dirk Farin <dirk.farin@gmail.com>
 *
 * This file is part of libde265.
 *
 * libde265 is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libde265 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with libde265.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "tests.h"

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "libde265/acceleration.h"
#include "libde265/fallback.h"
#include "libde265/fallback-dct.h"

#if HAVE_SSE4_1
#include "libde265/x86/sse-highbit.h"
#endif

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <sys/time.h>

namespace {

struct RNG {
  uint32_t s;
  explicit RNG(uint32_t seed) : s(seed ? seed : 1) {}
  uint32_t next() { s ^= s<<13; s ^= s>>17; s ^= s<<5; return s; }
  int range(int lo, int hi) { return lo + (int)(next() % (uint32_t)(hi-lo+1)); }
};

double now_sec() { struct timeval tv; gettimeofday(&tv,0); return tv.tv_sec + tv.tv_usec*1e-6; }

const int kMaxW = 64;
const int kMaxH = 64;
const int kPad = 16;

int compareBuffers(const char* tag, const uint16_t* a, const uint16_t* b, size_t n,
                   bool quiet)
{
  for (size_t i = 0; i < n; i++) {
    if (a[i] != b[i]) {
      if (!quiet) {
        printf("  MISMATCH %s at index %zu: ref=%d simd=%d\n", tag, i, a[i], b[i]);
      }
      return 0;
    }
  }
  return 1;
}

int compareBuffers16(const char* tag, const int16_t* a, const int16_t* b, size_t n,
                     bool quiet)
{
  for (size_t i = 0; i < n; i++) {
    if (a[i] != b[i]) {
      if (!quiet) {
        printf("  MISMATCH %s at index %zu: ref=%d simd=%d\n", tag, i, a[i], b[i]);
      }
      return 0;
    }
  }
  return 1;
}

#if HAVE_SSE4_1

int checkEpel(const acceleration_functions& ref, int w, int h, int depth,
              int mx, int my, RNG& rng, bool quiet)
{
  const int stride = kMaxW + 2*kPad;
  std::vector<uint16_t> src((kMaxH + 2*kPad)*stride);
  const int maxval = (1 << depth) - 1;
  for (auto& v : src) v = (uint16_t)rng.range(0, maxval);
  const uint16_t* p = src.data() + kPad*stride + kPad;

  const int dststride = w + 17;
  std::vector<int16_t> dstRef((size_t)h*dststride, 0);
  std::vector<int16_t> dstSimd((size_t)h*dststride, 0);
  alignas(16) int16_t mcbuf1[64*71];
  alignas(16) int16_t mcbuf2[64*71];

  if (mx == 0 && my == 0) {
    ref.put_hevc_epel(dstRef.data(), dststride, p, stride, w, h, mx, my, mcbuf1, depth);
    put_hevc_epel_16_sse4(dstSimd.data(), dststride, p, stride, w, h, mx, my, mcbuf2, depth);
  }
  else if (my == 0) {
    ref.put_hevc_epel_h(dstRef.data(), dststride, p, stride, w, h, mx, my, mcbuf1, depth);
    put_hevc_epel_h_16_sse4(dstSimd.data(), dststride, p, stride, w, h, mx, my, mcbuf2, depth);
  }
  else if (mx == 0) {
    ref.put_hevc_epel_v(dstRef.data(), dststride, p, stride, w, h, mx, my, mcbuf1, depth);
    put_hevc_epel_v_16_sse4(dstSimd.data(), dststride, p, stride, w, h, mx, my, mcbuf2, depth);
  }
  else {
    ref.put_hevc_epel_hv(dstRef.data(), dststride, p, stride, w, h, mx, my, mcbuf1, depth);
    put_hevc_epel_hv_16_sse4(dstSimd.data(), dststride, p, stride, w, h, mx, my, mcbuf2, depth);
  }

  return compareBuffers16("epel", dstRef.data(), dstSimd.data(), dstRef.size(), quiet);
}

int checkQpel(const acceleration_functions& ref, int w, int h, int depth,
              int xFrac, int yFrac, RNG& rng, bool quiet)
{
  const int stride = kMaxW + 2*kPad;
  std::vector<uint16_t> src((kMaxH + 2*kPad)*stride);
  const int maxval = (1 << depth) - 1;
  for (auto& v : src) v = (uint16_t)rng.range(0, maxval);
  const uint16_t* p = src.data() + kPad*stride + kPad;

  const int dststride = w + 17;
  std::vector<int16_t> dstRef((size_t)h*dststride, 0);
  std::vector<int16_t> dstSimd((size_t)h*dststride, 0);
  alignas(16) int16_t mcbuf1[64*71];
  alignas(16) int16_t mcbuf2[64*71];

  ref.put_hevc_qpel(dstRef.data(), dststride, p, stride, w, h, mcbuf1, xFrac, yFrac, depth);
  put_hevc_qpel_16_sse4(dstSimd.data(), dststride, p, stride, w, h, mcbuf2,
                        xFrac, yFrac, depth);

  return compareBuffers16("qpel", dstRef.data(), dstSimd.data(), dstRef.size(), quiet);
}

int checkWeighted(const acceleration_functions& ref, int w, int h, int depth,
                  RNG& rng, bool quiet)
{
  const int stride = kMaxW + 17;
  std::vector<int16_t> s1((size_t)h*stride), s2((size_t)h*stride);
  for (size_t i = 0; i < s1.size(); i++) {
    s1[i] = (int16_t)rng.range(-20000, 20000);
    s2[i] = (int16_t)rng.range(-20000, 20000);
  }

  std::vector<uint16_t> outRef((size_t)h*stride), outSimd((size_t)h*stride);

  bool ok = true;
  ref.put_unweighted_pred(outRef.data(), stride, s1.data(), stride, w, h, depth);
  put_unweighted_pred_16_sse4(outSimd.data(), stride, s1.data(), stride, w, h, depth);
  ok &= compareBuffers("unweighted", outRef.data(), outSimd.data(), outRef.size(), quiet);

  ref.put_weighted_pred_avg(outRef.data(), stride, s1.data(), s2.data(), stride, w, h, depth);
  put_weighted_pred_avg_16_sse4(outSimd.data(), stride, s1.data(), s2.data(), stride, w, h, depth);
  ok &= compareBuffers("weighted-avg", outRef.data(), outSimd.data(), outRef.size(), quiet);

  const int log2WD = 6 + std::max(2, 14-depth);
  const int o = rng.range(-256, 256);
  const int wgt = rng.range(0, 255);
  ref.put_weighted_pred(outRef.data(), stride, s1.data(), stride, w, h, wgt, o, log2WD, depth);
  put_weighted_pred_16_sse4(outSimd.data(), stride, s1.data(), stride, w, h, wgt, o, log2WD, depth);
  ok &= compareBuffers("weighted", outRef.data(), outSimd.data(), outRef.size(), quiet);

  const int o1 = rng.range(-256, 256);
  const int o2 = rng.range(-256, 256);
  const int w1 = rng.range(0, 255);
  const int w2 = rng.range(0, 255);
  ref.put_weighted_bipred(outRef.data(), stride, s1.data(), s2.data(), stride, w, h,
                          w1, o1, w2, o2, log2WD, depth);
  put_weighted_bipred_16_sse4(outSimd.data(), stride, s1.data(), s2.data(), stride, w, h,
                              w1, o1, w2, o2, log2WD, depth);
  ok &= compareBuffers("bipred", outRef.data(), outSimd.data(), outRef.size(), quiet);

  return ok;
}

int checkIntra(const acceleration_functions& ref, int nT, int depth, int mode,
               int cIdx, int disable, RNG& rng, bool quiet)
{
  const int borderMax = 64;
  std::vector<uint16_t> bmem(4*borderMax+1);
  const int maxval = (1 << depth) - 1;
  for (auto& v : bmem) v = (uint16_t)rng.range(0, maxval);
  const uint16_t* border = bmem.data() + 2*borderMax;

  const int stride = nT + 17;
  std::vector<uint16_t> dstRef((size_t)nT*stride), dstSimd((size_t)nT*stride);
  for (size_t i = 0; i < dstRef.size(); i++) {
    dstRef[i] = dstSimd[i] = (uint16_t)rng.range(0, maxval);
  }

  if (mode == 0) {
    ref.intra_pred_planar(dstRef.data(), stride, nT, cIdx, border);
    intra_pred_planar_16_sse4(dstSimd.data(), stride, nT, cIdx, border);
  }
  else if (mode == 1) {
    ref.intra_pred_dc(dstRef.data(), stride, nT, cIdx, border);
    intra_pred_dc_16_sse4(dstSimd.data(), stride, nT, cIdx, border);
  }
  else {
    ref.intra_pred_angular(dstRef.data(), stride, depth, disable, 0, 0, mode, nT, cIdx, border);
    intra_pred_angular_16_sse4(dstSimd.data(), stride, depth, disable, 0, 0, mode, nT, cIdx, border);
  }

  return compareBuffers("intra", dstRef.data(), dstSimd.data(), dstRef.size(), quiet);
}

int checkDeblock(const acceleration_functions& ref, int vertical, int dE, int dEp,
                 int dEq, int tc, int filterP, int filterQ, int depth, RNG& rng,
                 bool quiet)
{
  const int W = 24, H = 24, off = 8;
  const int maxval = (1 << depth) - 1;
  std::vector<uint16_t> a(W*H), b(W*H);
  for (int i = 0; i < W*H; i++) {
    a[i] = b[i] = (uint16_t)rng.range(0, maxval);
  }
  uint16_t* pa = a.data() + off*W + off;
  uint16_t* pb = b.data() + off*W + off;

  ref.deblock_luma_16(pa, W, vertical, dE, dEp, dEq, tc, filterP, filterQ, depth);
  deblock_luma_16_sse4(pb, W, vertical, dE, dEp, dEq, tc, filterP, filterQ, depth);

  return compareBuffers("deblock-luma", a.data(), b.data(), a.size(), quiet);
}

int checkTransforms(int nT, int depth, int scenario, RNG& rng, bool quiet)
{
  const int stride = nT + 17;
  alignas(16) int16_t coeffsRef[32*32];
  alignas(16) int16_t coeffsSimd[32*32];
  std::vector<uint16_t> dstRef((size_t)nT*stride), dstSimd((size_t)nT*stride);
  const int maxval = (1 << depth) - 1;

  for (int i = 0; i < nT*nT; i++) {
    int16_t c;
    if (scenario == 0) {
      c = (rng.next() & 3) ? 0 : (int16_t)rng.range(-512, 512);
    }
    else if (scenario == 1) {
      c = (int16_t)rng.range(-2048, 2048);
    }
    else {
      c = (int16_t)rng.range(-32768, 32767);
    }
    coeffsRef[i] = coeffsSimd[i] = c;
  }
  for (size_t i = 0; i < dstRef.size(); i++) {
    dstRef[i] = dstSimd[i] = (uint16_t)rng.range(0, maxval);
  }

  bool ok = true;
  switch (nT) {
    case 4:
      transform_4x4_add_16_fallback(dstRef.data(), coeffsRef, stride, depth);
      transform_4x4_add_16_sse4(dstSimd.data(), coeffsSimd, stride, depth);
      ok &= compareBuffers("idct4", dstRef.data(), dstSimd.data(), dstRef.size(), quiet);
      break;
    case 8:
      transform_8x8_add_16_fallback(dstRef.data(), coeffsRef, stride, depth);
      transform_8x8_add_16_sse4(dstSimd.data(), coeffsSimd, stride, depth);
      ok &= compareBuffers("idct8", dstRef.data(), dstSimd.data(), dstRef.size(), quiet);
      break;
    case 16:
      transform_16x16_add_16_fallback(dstRef.data(), coeffsRef, stride, depth);
      transform_16x16_add_16_sse4(dstSimd.data(), coeffsSimd, stride, depth);
      ok &= compareBuffers("idct16", dstRef.data(), dstSimd.data(), dstRef.size(), quiet);
      break;
    case 32:
      transform_32x32_add_16_fallback(dstRef.data(), coeffsRef, stride, depth);
      transform_32x32_add_16_sse4(dstSimd.data(), coeffsSimd, stride, depth);
      ok &= compareBuffers("idct32", dstRef.data(), dstSimd.data(), dstRef.size(), quiet);
      break;
  }

  // The 4x4 luma DST uses a separate entry point.
  if (nT == 4) {
    for (int i = 0; i < nT*nT; i++) {
      int16_t c;
      if (scenario == 0) {
        c = (rng.next() & 3) ? 0 : (int16_t)rng.range(-512, 512);
      }
      else if (scenario == 1) {
        c = (int16_t)rng.range(-2048, 2048);
      }
      else {
        c = (int16_t)rng.range(-32768, 32767);
      }
      coeffsRef[i] = coeffsSimd[i] = c;
    }
    for (size_t i = 0; i < dstRef.size(); i++) {
      dstRef[i] = dstSimd[i] = (uint16_t)rng.range(0, maxval);
    }
    transform_4x4_luma_add_16_fallback(dstRef.data(), coeffsRef, stride, depth);
    transform_4x4_dst_add_16_sse4(dstSimd.data(), coeffsSimd, stride, depth);
    ok &= compareBuffers("idst4", dstRef.data(), dstSimd.data(), dstRef.size(), quiet);
  }

  return ok;
}

#endif // HAVE_SSE4_1

} // namespace


class HighBitSseTest : public Test
{
public:
  const char* getName() const { return "highbit-simd"; }
  const char* getDescription() const {
    return "compare 10-bit SSE kernels against scalar fallback and benchmark";
  }

  bool work(bool quiet) {
#if HAVE_SSE4_1
    RNG rng(0x10B175E);
    acceleration_functions ref;
    init_acceleration_functions_fallback(&ref);

    bool ok = true;
    int nCases = 0;

    // --- motion compensation: both chroma and luma fractional positions ---
    const int sizes[] = { 2, 4, 8, 16, 32, 64 };
    for (int depth = 9; depth <= 12; depth++) {
      for (int w : sizes) {
        for (int h : sizes) {
          if (w > kMaxW || h > kMaxH) continue;
          for (int mx = 0; mx < 8; mx++) {
            for (int my = 0; my < 8; my++) {
              nCases++;
              ok &= checkEpel(ref, w, h, depth, mx, my, rng, quiet);
            }
          }
          for (int xf = 0; xf < 4; xf++) {
            for (int yf = 0; yf < 4; yf++) {
              nCases++;
              ok &= checkQpel(ref, w, h, depth, xf, yf, rng, quiet);
            }
          }
        }
      }
    }

    // --- weighted prediction ---
    for (int depth = 9; depth <= 12; depth++) {
      for (int w = 2; w <= kMaxW; w += 2) {
        nCases++;
        ok &= checkWeighted(ref, w, 8, depth, rng, quiet);
      }
    }

    // --- intra prediction: all modes, sizes, components, filter flags ---
    const int intraSizes[] = { 4, 8, 16, 32, 64 };
    const int intraDepths[] = { 9, 10, 11, 12, 16 };
    for (int depth : intraDepths) {
      for (int nT : intraSizes) {
        for (int mode = 0; mode <= 34; mode++) {
          for (int cIdx = 0; cIdx <= 1; cIdx++) {
            for (int disable = 0; disable <= 1; disable++) {
              nCases++;
              ok &= checkIntra(ref, nT, depth, mode, cIdx, disable, rng, quiet);
            }
          }
        }
      }
    }

    // --- deblocking ---
    for (int depth = 9; depth <= 16; depth++) {
      for (int rep = 0; rep < 20; rep++) {
        for (int vert = 0; vert < 2; vert++) {
          const int tc = rng.range(1, 24);
          for (int dE = 1; dE <= 2; dE++)
            for (int dEp = 0; dEp < 2; dEp++)
              for (int dEq = 0; dEq < 2; dEq++)
                for (int fP = 0; fP < 2; fP++)
                  for (int fQ = 0; fQ < 2; fQ++) {
                    nCases++;
                    ok &= checkDeblock(ref, vert, dE, dEp, dEq, tc, fP, fQ,
                                       depth, rng, quiet);
                  }
        }
      }
    }

    // --- inverse transforms ---
    const int transformSizes[] = { 4, 8, 16, 32 };
    for (int depth = 9; depth <= 12; depth++) {
      for (int nT : transformSizes) {
        for (int scen = 0; scen < 3; scen++) {
          for (int rep = 0; rep < 4; rep++) {
            nCases++;
            ok &= checkTransforms(nT, depth, scen, rng, quiet);
          }
        }
      }
    }

    if (!quiet) {
      printf("ran %d high-bit-depth comparison cases: %s\n\n",
             nCases, ok ? "all passed" : "FAILURES");

      // --- benchmarks ---
      printf("high-bit-depth (10 bit) timing:\n");

      const int stride = kMaxW + 2*kPad;
      std::vector<uint16_t> src((kMaxH + 2*kPad)*stride);
      for (auto& v : src) v = (uint16_t)rng.range(0, 1023);
      const uint16_t* p = src.data() + kPad*stride + kPad;
      std::vector<int16_t> out((size_t)kMaxH*(kMaxW+17));
      alignas(16) int16_t mcbuf[64*71];

      auto timeit = [](auto fn, int iters) {
        fn();
        double t0 = now_sec();
        for (int i = 0; i < iters; i++) fn();
        double t1 = now_sec();
        return (t1-t0) / iters * 1e9;
      };

      for (int sz : { 8, 16, 32, 64 }) {
        double s = timeit([&]{ ref.put_hevc_qpel(out.data(), kMaxW+17, p, stride,
                                                sz, sz, mcbuf, 1, 1, 10); }, 20000);
        double e = timeit([&]{ put_hevc_qpel_16_sse4(out.data(), kMaxW+17, p, stride,
                                                     sz, sz, mcbuf, 1, 1, 10); }, 20000);
        printf("  qpel %2dx%-2d  scalar=%8.1f ns  sse=%8.1f ns  x%.2f\n",
               sz, sz, s, e, s/e);
      }
      for (int sz : { 8, 16, 32, 64 }) {
        double s = timeit([&]{ ref.put_hevc_epel_hv(out.data(), kMaxW+17, p, stride,
                                                   sz, sz, 1, 1, mcbuf, 10); }, 20000);
        double e = timeit([&]{ put_hevc_epel_hv_16_sse4(out.data(), kMaxW+17, p, stride,
                                                        sz, sz, 1, 1, mcbuf, 10); }, 20000);
        printf("  epel %2dx%-2d  scalar=%8.1f ns  sse=%8.1f ns  x%.2f\n",
               sz, sz, s, e, s/e);
      }

      for (int sz : { 4, 8, 16, 32 }) {
        alignas(16) int16_t coeffs[32*32];
        std::vector<uint16_t> dst((size_t)sz*(sz+17));
        for (int i = 0; i < sz*sz; i++) coeffs[i] = (int16_t)rng.range(-2048, 2048);

        auto refFn = [&] {
          if (sz == 4) transform_4x4_add_16_fallback(dst.data(), coeffs, sz+17, 10);
          else if (sz == 8) transform_8x8_add_16_fallback(dst.data(), coeffs, sz+17, 10);
          else if (sz == 16) transform_16x16_add_16_fallback(dst.data(), coeffs, sz+17, 10);
          else transform_32x32_add_16_fallback(dst.data(), coeffs, sz+17, 10);
        };
        auto sseFn = [&] {
          if (sz == 4) transform_4x4_add_16_sse4(dst.data(), coeffs, sz+17, 10);
          else if (sz == 8) transform_8x8_add_16_sse4(dst.data(), coeffs, sz+17, 10);
          else if (sz == 16) transform_16x16_add_16_sse4(dst.data(), coeffs, sz+17, 10);
          else transform_32x32_add_16_sse4(dst.data(), coeffs, sz+17, 10);
        };

        double s = timeit(refFn, 20000);
        double e = timeit(sseFn, 20000);
        printf("  idct %2dx%-2d  scalar=%8.1f ns  sse=%8.1f ns  x%.2f\n",
               sz, sz, s, e, s/e);
      }

      {
        alignas(16) int16_t coeffs[4*4];
        std::vector<uint16_t> dst(4*(4+17));
        for (int i = 0; i < 16; i++) coeffs[i] = (int16_t)rng.range(-2048, 2048);
        double s = timeit([&]{ transform_4x4_luma_add_16_fallback(dst.data(), coeffs, 21, 10); },
                          20000);
        double e = timeit([&]{ transform_4x4_dst_add_16_sse4(dst.data(), coeffs, 21, 10); },
                          20000);
        printf("  idst  4x4   scalar=%8.1f ns  sse=%8.1f ns  x%.2f\n", s, e, s/e);
      }

      {
        std::vector<int16_t> pred1((size_t)16*16), pred2((size_t)16*16);
        for (size_t i = 0; i < pred1.size(); i++) {
          pred1[i] = (int16_t)rng.range(-12000, 12000);
          pred2[i] = (int16_t)rng.range(-12000, 12000);
        }
        std::vector<uint16_t> dst((size_t)16*16);
        double s = timeit([&]{ ref.put_weighted_bipred(dst.data(), 16, pred1.data(),
                                                       pred2.data(), 16, 16, 16, 64, 64,
                                                       -64, 64, 7, 10); }, 50000);
        double e = timeit([&]{ put_weighted_bipred_16_sse4(dst.data(), 16, pred1.data(),
                                                           pred2.data(), 16, 16, 16, 64, 64,
                                                           -64, 64, 7, 10); }, 50000);
        printf("  bipred 16x16 scalar=%8.1f ns  sse=%8.1f ns  x%.2f\n", s, e, s/e);
      }

      {
        std::vector<uint16_t> border(4*64+1);
        for (auto& v : border) v = (uint16_t)rng.range(0, 1023);
        const uint16_t* b = border.data() + 2*64;
        std::vector<uint16_t> dst((size_t)16*33);
        double ps = timeit([&]{ ref.intra_pred_planar(dst.data(), 33, 16, 0, b); }, 50000);
        double pe = timeit([&]{ intra_pred_planar_16_sse4(dst.data(), 33, 16, 0, b); }, 50000);
        double ds = timeit([&]{ ref.intra_pred_dc(dst.data(), 33, 16, 0, b); }, 50000);
        double de = timeit([&]{ intra_pred_dc_16_sse4(dst.data(), 33, 16, 0, b); }, 50000);
        printf("  intra planar scalar=%8.1f ns  sse=%8.1f ns  x%.2f\n", ps, pe, ps/pe);
        printf("  intra DC     scalar=%8.1f ns  sse=%8.1f ns  x%.2f\n", ds, de, ds/de);
      }

      {
        std::vector<uint16_t> edge((size_t)24*24);
        for (auto& v : edge) v = (uint16_t)rng.range(0, 1023);
        uint16_t* e = edge.data() + 8*24 + 8;
        double ls = timeit([&]{ ref.deblock_luma_16(e, 24, 1, 2, 1, 1, 12, 1, 1, 10); }, 500000);
        double le = timeit([&]{ deblock_luma_16_sse4(e, 24, 1, 2, 1, 1, 12, 1, 1, 10); }, 500000);
        printf("  deblock luma scalar=%8.1f ns  sse=%8.1f ns  x%.2f\n", ls, le, ls/le);
      }
    }

    return ok;
#else
    if (!quiet) printf("SSE4.1 not available -- skipping high-bit-depth SIMD test.\n");
    return true;
#endif
  }
} highBitSseTest;
