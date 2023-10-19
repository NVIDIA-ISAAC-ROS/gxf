/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <utility>

#include "gxf/std/timestamp.hpp"

namespace nvidia {
namespace gxf {

Expected<Timestamp> getTimestamp(MultiSourceTimestamp const& timestamps,
                                 TimeDomainID const& time_domain_id) {
  for (size_t i = 0; i < timestamps.size(); i++) {
    const std::pair<Timestamp, TimeDomainID> timestamp = timestamps.at(i).value();
    if (timestamp.second == time_domain_id) {
      return timestamp.first;
    }
  }
  return Unexpected{GXF_QUERY_NOT_FOUND};
}

}  // namespace gxf
}  // namespace nvidia

