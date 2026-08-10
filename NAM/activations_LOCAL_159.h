#pragma once

#include <cassert>
#include <cmath> // expf
#include <iostream> // std::cerr (kept for potential debug use)
#include <stdexcept> // std::invalid_argument
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Dense>

#include "json.hpp"

// SIMD detection for activation vectorization (SSE2 is baseline for x64)
#if defined(_M_X64) || defined(_M_AMD64) || defined(__x86_64__) || defined(__SSE2__)
#define NAM_SIMD_SSE2 1
#include <immintrin.h>
#endif

namespace nam
{
namespace activations
{

// Forward declaration
class Activation;

// Strongly-typed activation type enum
enum class ActivationType
{
  Tanh,
  Hardtanh,
  Fasttanh,
  ReLU,
  LeakyReLU,
  PReLU,
  Sigmoid,
  SiLU, // aka Swish
  Hardswish,
  LeakyHardtanh,
  Softsign
};

// Strongly-typed activation configuration
struct ActivationConfig
{
  ActivationType type;

  // Optional parameters (used by specific activation types)
  std::optional<float> negative_slope; // LeakyReLU, PReLU (single)
  std::optional<std::vector<float>> negative_slopes; // PReLU (per-channel)
  std::optional<float> min_val; // LeakyHardtanh
  std::optional<float> max_val; // LeakyHardtanh
  std::optional<float> min_slope; // LeakyHardtanh
  std::optional<float> max_slope; // LeakyHardtanh

  // Convenience constructors
  static ActivationConfig simple(ActivationType t);
  static ActivationConfig from_json(const nlohmann::json& j);
};
inline float relu(float x)
{
  return x > 0.0f ? x : 0.0f;
};

inline float sigmoid(float x)
{
  return 1.0f / (1.0f + expf(-x));
};

inline float hard_tanh(float x)
{
  const float t = x < -1 ? -1 : x;
  return t > 1 ? 1 : t;
}

inline float leaky_hardtanh(float x, float min_val, float max_val, float min_slope, float max_slope)
{
  if (x < min_val)
  {
    return (x - min_val) * min_slope + min_val;
  }
  else if (x > max_val)
  {
    return (x - max_val) * max_slope + max_val;
  }
  else
  {
    return x;
  }
}

inline float fast_tanh(const float x)
{
  const float ax = fabsf(x);
  const float x2 = x * x;

  return (x * (2.45550750702956f + 2.45550750702956f * ax + (0.893229853513558f + 0.821226666969744f * ax) * x2)
          / (2.44506634652299f + (2.44506634652299f + x2) * fabsf(x + 0.814642734961073f * x * ax)));
}

inline float fast_sigmoid(const float x)
{
  return 0.5f * (fast_tanh(x * 0.5f) + 1.0f);
}

inline float leaky_relu(float x, float negative_slope)
{
  return x > 0.0f ? x : negative_slope * x;
}
inline float leaky_relu(float x)
{
  return leaky_relu(x, 0.01);
}


inline float swish(float x)
{
  return x * sigmoid(x);
}

inline float hardswish(float x)
{
  // Branchless implementation using clamp
  // hardswish(x) = x * relu6(x + 3) / 6
  //              = x * clamp(x + 3, 0, 6) / 6
  const float t = x + 3.0f;
  const float clamped = t < 0.0f ? 0.0f : (t > 6.0f ? 6.0f : t);
  return x * clamped * (1.0f / 6.0f);
}

inline float softsign(float x)
{
  return x / (1.0f + fabsf(x));
}

// ==========================================================================
// SSE2 SIMD helper functions — process 4 floats at a time.
// Numerically identical to scalar helpers (no approximation changes).
// ==========================================================================

#ifdef NAM_SIMD_SSE2
namespace simd
{

// Absolute value: clear sign bit
inline __m128 abs_ps(__m128 x)
{
  return _mm_andnot_ps(_mm_set1_ps(-0.0f), x);
}

// ReLU: max(x, 0)
inline __m128 relu_ps(__m128 x)
{
  return _mm_max_ps(x, _mm_setzero_ps());
}

// HardTanh: clamp(x, -1, 1)
inline __m128 hard_tanh_ps(__m128 x)
{
  return _mm_max_ps(_mm_min_ps(x, _mm_set1_ps(1.0f)), _mm_set1_ps(-1.0f));
}

// FastTanh: vectorized rational polynomial (same coefficients as scalar fast_tanh)
inline __m128 fast_tanh_ps(__m128 x)
{
  const __m128 ax = abs_ps(x);
  const __m128 x2 = _mm_mul_ps(x, x);

  const __m128 c1 = _mm_set1_ps(2.45550750702956f);
  const __m128 c2 = _mm_set1_ps(0.893229853513558f);
  const __m128 c3 = _mm_set1_ps(0.821226666969744f);
  const __m128 c4 = _mm_set1_ps(2.44506634652299f);
  const __m128 c5 = _mm_set1_ps(0.814642734961073f);

  // numerator = x * (c1 + c1*ax + (c2 + c3*ax) * x2)
  __m128 inner = _mm_add_ps(c2, _mm_mul_ps(c3, ax));
  inner = _mm_mul_ps(inner, x2);
  __m128 num = _mm_add_ps(c1, _mm_add_ps(_mm_mul_ps(c1, ax), inner));
  num = _mm_mul_ps(x, num);

  // denominator = c4 + (c4 + x2) * |x + c5 * x * ax|
  __m128 x_ax = _mm_mul_ps(x, ax);
  __m128 denom_inner = _mm_add_ps(x, _mm_mul_ps(c5, x_ax));
  denom_inner = abs_ps(denom_inner);
  __m128 denom = _mm_add_ps(c4, _mm_mul_ps(_mm_add_ps(c4, x2), denom_inner));

  return _mm_div_ps(num, denom);
}

// LeakyReLU: x > 0 ? x : slope * x
inline __m128 leaky_relu_ps(__m128 x, __m128 slope)
{
  __m128 mask = _mm_cmpgt_ps(x, _mm_setzero_ps());
  __m128 neg = _mm_mul_ps(slope, x);
  return _mm_or_ps(_mm_and_ps(mask, x), _mm_andnot_ps(mask, neg));
}

// HardSwish: x * clamp(x+3, 0, 6) / 6
inline __m128 hardswish_ps(__m128 x)
{
  const __m128 three = _mm_set1_ps(3.0f);
  const __m128 six = _mm_set1_ps(6.0f);
  const __m128 inv6 = _mm_set1_ps(1.0f / 6.0f);
  __m128 t = _mm_add_ps(x, three);
  t = _mm_max_ps(_mm_min_ps(t, six), _mm_setzero_ps());
  return _mm_mul_ps(_mm_mul_ps(x, t), inv6);
}

// Softsign: x / (1 + |x|)
inline __m128 softsign_ps(__m128 x)
{
  return _mm_div_ps(x, _mm_add_ps(_mm_set1_ps(1.0f), abs_ps(x)));
}

// LeakyHardTanh: piecewise linear with slopes outside [min_val, max_val]
inline __m128 leaky_hardtanh_ps(__m128 x, __m128 vmin, __m128 vmax, __m128 min_slope, __m128 max_slope)
{
  __m128 mask_lo = _mm_cmplt_ps(x, vmin);
  __m128 mask_hi = _mm_cmpgt_ps(x, vmax);

  __m128 lo_result = _mm_add_ps(_mm_mul_ps(_mm_sub_ps(x, vmin), min_slope), vmin);
  __m128 hi_result = _mm_add_ps(_mm_mul_ps(_mm_sub_ps(x, vmax), max_slope), vmax);

  // Start with x (mid case), blend in lo/hi results where masks are active
  __m128 result = _mm_or_ps(_mm_andnot_ps(mask_lo, x), _mm_and_ps(mask_lo, lo_result));
  result = _mm_or_ps(_mm_andnot_ps(mask_hi, result), _mm_and_ps(mask_hi, hi_result));
  return result;
}

} // namespace simd
#endif // NAM_SIMD_SSE2

class Activation
{
public:
  // Type alias for shared pointer to Activation
  using Ptr = std::shared_ptr<Activation>;

  Activation() = default;
  virtual ~Activation() = default;
  virtual void apply(Eigen::MatrixXf& matrix) { apply(matrix.data(), matrix.rows() * matrix.cols()); }
  virtual void apply(Eigen::Block<Eigen::MatrixXf> block)
  {
    // Block must be contiguous in memory (outerStride == rows) for flat data() access.
    // Non-contiguous blocks (e.g. topRows() of a wider matrix) would read/write wrong elements.
    assert(block.outerStride() == block.rows());
    apply(block.data(), block.rows() * block.cols());
  }
  virtual void apply(Eigen::Block<Eigen::MatrixXf, -1, -1, true> block)
  {
    // Inner-panel blocks (e.g. leftCols()) are always contiguous for column-major matrices,
    // but assert anyway for safety.
    assert(block.outerStride() == block.rows());
    apply(block.data(), block.rows() * block.cols());
  }
  virtual void apply(float* data, long size) = 0;

  static Ptr get_activation(const std::string name);
  static Ptr get_activation(const ActivationConfig& config);
  static Ptr get_activation(const nlohmann::json& activation_config);
  static void enable_fast_tanh();
  static void disable_fast_tanh();
  static bool using_fast_tanh;
  static void enable_lut(std::string function_name, float min, float max, std::size_t n_points);
  static void disable_lut(std::string function_name);

protected:
  static std::unordered_map<std::string, Ptr> _activations;
};

// identity function activation--"do nothing"
class ActivationIdentity : public nam::activations::Activation
{
public:
  ActivationIdentity() = default;
  ~ActivationIdentity() = default;
  virtual void apply(float* data, long size) override {};
};

// Tanh — uses std::tanh (transcendental); left scalar.
// Users wanting speed should use ActivationFastTanh which has a SIMD path.
class ActivationTanh : public Activation
{
public:
  void apply(float* data, long size) override
  {
    for (long pos = 0; pos < size; pos++)
    {
      data[pos] = std::tanh(data[pos]);
    }
  }
};

class ActivationHardTanh : public Activation
{
public:
  void apply(float* data, long size) override
  {
#ifdef NAM_SIMD_SSE2
    long pos = 0;
    for (; pos + 3 < size; pos += 4)
    {
      __m128 v = _mm_loadu_ps(data + pos);
      _mm_storeu_ps(data + pos, simd::hard_tanh_ps(v));
    }
    for (; pos < size; pos++)
      data[pos] = hard_tanh(data[pos]);
#else
    for (long pos = 0; pos < size; pos++)
    {
      data[pos] = hard_tanh(data[pos]);
    }
#endif
  }
};

class ActivationLeakyHardTanh : public Activation
{
public:
  ActivationLeakyHardTanh() = default;
  ActivationLeakyHardTanh(float min_val_, float max_val_, float min_slope_, float max_slope_)
  {
    min_val = min_val_;
    max_val = max_val_;
    min_slope = min_slope_;
    max_slope = max_slope_;
  }
  void apply(float* data, long size) override
  {
#ifdef NAM_SIMD_SSE2
    const __m128 vmin = _mm_set1_ps(min_val);
    const __m128 vmax = _mm_set1_ps(max_val);
    const __m128 vmin_slope = _mm_set1_ps(min_slope);
    const __m128 vmax_slope = _mm_set1_ps(max_slope);
    long pos = 0;
    for (; pos + 3 < size; pos += 4)
    {
      __m128 v = _mm_loadu_ps(data + pos);
      _mm_storeu_ps(data + pos, simd::leaky_hardtanh_ps(v, vmin, vmax, vmin_slope, vmax_slope));
    }
    for (; pos < size; pos++)
      data[pos] = leaky_hardtanh(data[pos], min_val, max_val, min_slope, max_slope);
#else
    for (long pos = 0; pos < size; pos++)
    {
      data[pos] = leaky_hardtanh(data[pos], min_val, max_val, min_slope, max_slope);
    }
#endif
  }

private:
  float min_val = -1.0;
  float max_val = 1.0;
  float min_slope = 0.01;
  float max_slope = 0.01;
};

class ActivationFastTanh : public Activation
{
public:
  void apply(float* data, long size) override
  {
#ifdef NAM_SIMD_SSE2
    long pos = 0;
    for (; pos + 3 < size; pos += 4)
    {
      __m128 v = _mm_loadu_ps(data + pos);
      _mm_storeu_ps(data + pos, simd::fast_tanh_ps(v));
    }
    for (; pos < size; pos++)
      data[pos] = fast_tanh(data[pos]);
#else
    for (long pos = 0; pos < size; pos++)
    {
      data[pos] = fast_tanh(data[pos]);
    }
#endif
  }
};

class ActivationReLU : public Activation
{
public:
  void apply(float* data, long size) override
  {
#ifdef NAM_SIMD_SSE2
    long pos = 0;
    for (; pos + 3 < size; pos += 4)
    {
      __m128 v = _mm_loadu_ps(data + pos);
      _mm_storeu_ps(data + pos, simd::relu_ps(v));
    }
    for (; pos < size; pos++)
      data[pos] = relu(data[pos]);
#else
    for (long pos = 0; pos < size; pos++)
      data[pos] = relu(data[pos]);
#endif
  }
};

class ActivationLeakyReLU : public Activation
{
public:
  ActivationLeakyReLU() = default;
  ActivationLeakyReLU(float ns) { negative_slope = ns; }
  void apply(float* data, long size) override
  {
#ifdef NAM_SIMD_SSE2
    const __m128 slope = _mm_set1_ps(negative_slope);
    long pos = 0;
    for (; pos + 3 < size; pos += 4)
    {
      __m128 v = _mm_loadu_ps(data + pos);
      _mm_storeu_ps(data + pos, simd::leaky_relu_ps(v, slope));
    }
    for (; pos < size; pos++)
      data[pos] = leaky_relu(data[pos], negative_slope);
#else
    for (long pos = 0; pos < size; pos++)
    {
      data[pos] = leaky_relu(data[pos], negative_slope);
    }
#endif
  }

private:
  float negative_slope = 0.01;
};

// PReLU — per-channel slopes with small channel counts (2-8) don't pack well
// into SIMD lanes. The stride-based loop already eliminates the modulo overhead.
class ActivationPReLU : public Activation
{
public:
  ActivationPReLU() = default;
  ActivationPReLU(float ns)
  {
    negative_slopes.clear();
    negative_slopes.push_back(ns);
  }
  ActivationPReLU(std::vector<float> ns) { negative_slopes = ns; }

  void apply(float* data, long size) override
  {
    // Avoid modulo in hot loop: stride over channels explicitly
    const long num_ch = (long)negative_slopes.size();
    const long num_frames = size / num_ch;
    const float* slopes = negative_slopes.data();
    for (long f = 0; f < num_frames; f++)
    {
      float* col = data + f * num_ch;
      for (long c = 0; c < num_ch; c++)
      {
        col[c] = leaky_relu(col[c], slopes[c]);
      }
    }
  }

  void apply(Eigen::MatrixXf& matrix) override
  {
    // Matrix is organized as (channels, time_steps)
    unsigned long actual_channels = static_cast<unsigned long>(matrix.rows());

#ifndef NDEBUG
    if (actual_channels != negative_slopes.size())
    {
      throw std::invalid_argument("PReLU: Received " + std::to_string(actual_channels)
                                  + " channels, but activation has " + std::to_string(negative_slopes.size())
                                  + " channels");
    }
#endif

    // Apply each negative slope to its corresponding channel
    for (unsigned long channel = 0; channel < actual_channels; channel++)
    {
      // Apply the negative slope to all time steps in this channel
      for (int time_step = 0; time_step < matrix.cols(); time_step++)
      {
        matrix(channel, time_step) = leaky_relu(matrix(channel, time_step), negative_slopes[channel]);
      }
    }
  }

private:
  std::vector<float> negative_slopes;
};


// Sigmoid — uses transcendental expf; left scalar.
class ActivationSigmoid : public Activation
{
public:
  void apply(float* data, long size) override
  {
    for (long pos = 0; pos < size; pos++)
      data[pos] = sigmoid(data[pos]);
  }
};

// Swish (SiLU) — depends on expf via sigmoid; left scalar.
class ActivationSwish : public Activation
{
public:
  void apply(float* data, long size) override
  {
    for (long pos = 0; pos < size; pos++)
      data[pos] = swish(data[pos]);
  }
};

class ActivationHardSwish : public Activation
{
public:
  void apply(float* data, long size) override
  {
#ifdef NAM_SIMD_SSE2
    long pos = 0;
    for (; pos + 3 < size; pos += 4)
    {
      __m128 v = _mm_loadu_ps(data + pos);
      _mm_storeu_ps(data + pos, simd::hardswish_ps(v));
    }
    for (; pos < size; pos++)
      data[pos] = hardswish(data[pos]);
#else
    for (long pos = 0; pos < size; pos++)
      data[pos] = hardswish(data[pos]);
#endif
  }
};

class ActivationSoftsign : public Activation
{
public:
  void apply(float* data, long size) override
  {
#ifdef NAM_SIMD_SSE2
    long pos = 0;
    for (; pos + 3 < size; pos += 4)
    {
      __m128 v = _mm_loadu_ps(data + pos);
      _mm_storeu_ps(data + pos, simd::softsign_ps(v));
    }
    for (; pos < size; pos++)
      data[pos] = softsign(data[pos]);
#else
    for (long pos = 0; pos < size; pos++)
      data[pos] = softsign(data[pos]);
#endif
  }
};

// FastLUTActivation — table lookup is inherently scalar (no gather until AVX2)
class FastLUTActivation : public Activation
{
public:
  FastLUTActivation(float min_x, float max_x, std::size_t size, std::function<float(float)> f)
  : min_x_(min_x)
  , max_x_(max_x)
  , size_(size)
  {

    step_ = (max_x - min_x) / (size - 1);
    inv_step_ = 1.0f / step_;
    table_.reserve(size);

    for (std::size_t i = 0; i < size; ++i)
    {
      table_.push_back(f(min_x + i * step_));
    }
  }

  // Fast lookup with linear interpolation
  inline float lookup(float x) const
  {
    // Clamp input to range (inline to avoid header dependency)
    x = x < min_x_ ? min_x_ : (x > max_x_ ? max_x_ : x);

    // Calculate float index
    float f_idx = (x - min_x_) * inv_step_;
    std::size_t i = static_cast<std::size_t>(f_idx);

    // Handle edge case at max_x_
    if (i >= size_ - 1)
      return table_.back();

    // Linear interpolation: y = y0 + (y1 - y0) * fractional_part
    float frac = f_idx - static_cast<float>(i);
    return table_[i] + (table_[i + 1] - table_[i]) * frac;
  }

  // Override base class virtual method to apply LUT lookup to array of floats
  void apply(float* data, long size) override
  {
    for (long i = 0; i < size; i++)
    {
      data[i] = lookup(data[i]);
    }
  }

private:
  float min_x_, max_x_, step_, inv_step_;
  size_t size_;
  std::vector<float> table_;
};

}; // namespace activations
}; // namespace nam
