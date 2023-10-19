/*
Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "gxf/std/metric.hpp"

#include <algorithm>
#include <limits>

namespace nvidia {
namespace gxf {

gxf_result_t Metric::registerInterface(Registrar* registrar) {
  Expected<void> result;
  result &= registrar->parameter(
      aggregation_policy_, "aggregation_policy", "Aggregation Policy",
      "Aggregation policy used to aggregate individual metric samples. Choices:"
      "{mean, min, max}.", Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(
      lower_threshold_, "lower_threshold", "Lower threshold",
      "Lower threshold of the metric's expected range",
      Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  result &= registrar->parameter(
      upper_threshold_, "upper_threshold", "Upper threshold",
      "Upper threshold of the metric's expected range",
      Registrar::NoDefaultParameter(), GXF_PARAMETER_FLAGS_OPTIONAL);
  return ToResultCode(result);
}

gxf_result_t Metric::initialize() {
  // Set the aggregation function based on the aggregation policy parameter, if provided.
  const auto maybe_aggregation_policy = aggregation_policy_.try_get();
  if (maybe_aggregation_policy) {
    if (maybe_aggregation_policy.value() == "mean") {
      setMeanAggregationFunction();
    } else if (maybe_aggregation_policy.value() == "root_mean_square") {
      setRootMeanSquareAggregationFunction();
    } else if (maybe_aggregation_policy.value() == "abs_max") {
      setAbsMaxAggregationFunction();
    } else if (maybe_aggregation_policy.value() == "max") {
      setMaxAggregationFunction();
    } else if (maybe_aggregation_policy.value() == "min") {
      setMinAggregationFunction();
    } else if (maybe_aggregation_policy.value() == "sum") {
      setSumAggregationFunction();
    } else if (maybe_aggregation_policy.value() == "fixed") {
      setFixedAggregationFunction();
    } else {
      GXF_LOG_ERROR(
          "Invalid aggregation policy. Choices are {mean, root_mean_square, abs_max, max, min, "
          "sum, fixed}. Use setAggregationFunction() to set a custom function.");
      return GXF_PARAMETER_OUT_OF_RANGE;
    }
  }
  return GXF_SUCCESS;
}

Expected<void> Metric::record(double sample) {
  if (aggregation_function_ == nullptr) {
    GXF_LOG_ERROR("Aggregation function must be set in order to record a sample to this metric.");
    return Unexpected(GXF_FAILURE);
  }
  aggregated_value_ = aggregation_function_(sample);
  return Success;
}

Expected<void> Metric::setAggregationFunction(aggregation_function_t aggregation_function) {
  if (aggregation_function_ != nullptr) {
    GXF_LOG_WARNING("Aggregation function has already been set.");
    return Unexpected(GXF_FAILURE);
  }
  aggregation_function_ = aggregation_function;
  return Success;
}

Expected<bool> Metric::evaluateSuccess() {
  if (!aggregated_value_) {
    return Unexpected(GXF_FAILURE);
  }

  const auto maybe_lower_threshold = lower_threshold_.try_get();
  const auto maybe_upper_threshold = upper_threshold_.try_get();

  // If neither threshold is set, return success
  if (!maybe_lower_threshold && !maybe_upper_threshold) {
    return true;
  }

  // If both are set, check that lower <= upper
  if (maybe_lower_threshold && maybe_upper_threshold) {
    if (maybe_lower_threshold.value() > maybe_upper_threshold.value()) {
      GXF_LOG_ERROR("Lower threshold must be less than or equal to the upper threshold.");
      return Unexpected(GXF_PARAMETER_OUT_OF_RANGE);
    }
  }

  bool result = true;
  if (maybe_lower_threshold) {
    result &= (aggregated_value_.value() >= maybe_lower_threshold.value());
  }
  if (maybe_upper_threshold) {
    result &= (aggregated_value_.value() <= maybe_upper_threshold.value());
  }
  return result;
}

Expected<double> Metric::getAggregatedValue() {
  if (!aggregated_value_) {
    return Unexpected(GXF_FAILURE);
  }
  return aggregated_value_.value();
}

Expected<double> Metric::getLowerThreshold() {
  const auto maybe_lower_threshold = lower_threshold_.try_get();
  if (!maybe_lower_threshold) {
    return Unexpected(GXF_NULL_POINTER);
  }
  return maybe_lower_threshold.value();
}

Expected<double> Metric::getUpperThreshold() {
  const auto maybe_upper_threshold = upper_threshold_.try_get();
  if (!maybe_upper_threshold) {
    return Unexpected(GXF_NULL_POINTER);
  }
  return maybe_upper_threshold.value();
}

Expected<void> Metric::setMeanAggregationFunction() {
  return setAggregationFunction([sample_count = 0.0, running_sum = 0.0](double sample) mutable {
        running_sum += sample;
        sample_count++;
        return running_sum / sample_count;
      });
}

Expected<void> Metric::setRootMeanSquareAggregationFunction() {
  return setAggregationFunction([sample_count = 0.0, running_mse = 0.0](double sample) mutable {
        running_mse += sample * sample;
        sample_count++;
        return std::sqrt(running_mse / sample_count);
      });
}

Expected<void> Metric::setAbsMaxAggregationFunction() {
  return setAggregationFunction(
      [aggregated_value = std::numeric_limits<double>::lowest()](double sample) mutable {
        aggregated_value = std::max(aggregated_value, std::abs(sample));
        return aggregated_value;
      });
}

Expected<void> Metric::setMaxAggregationFunction() {
  return setAggregationFunction(
      [aggregated_value = std::numeric_limits<double>::lowest()](double sample) mutable {
        aggregated_value = std::max(aggregated_value, sample);
        return aggregated_value;
      });
}

Expected<void> Metric::setMinAggregationFunction() {
  return setAggregationFunction(
      [aggregated_value = std::numeric_limits<double>::max()](double sample) mutable {
        aggregated_value = std::min(aggregated_value, sample);
        return aggregated_value;
      });
}

Expected<void> Metric::setSumAggregationFunction() {
  return setAggregationFunction([aggregated_value = 0.0](double sample) mutable {
    aggregated_value += sample;
    return aggregated_value;
  });
}

Expected<void> Metric::setFixedAggregationFunction() {
  return setAggregationFunction([](double sample) mutable {
    return sample;
  });
}

Expected<void> Metric::reset() {
  aggregated_value_ = Unexpected{GXF_UNINITIALIZED_VALUE};
  return Success;
}

}  // namespace gxf
}  // namespace nvidia
