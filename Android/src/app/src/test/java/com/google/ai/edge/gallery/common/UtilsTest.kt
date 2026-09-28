/*
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.ai.edge.gallery.common

import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner
import org.robolectric.shadows.ShadowBuild

@RunWith(RobolectricTestRunner::class)
class UtilsTest {

  @Test
  fun isAICoreSupported_nullAllowedDeviceModels_returnsFalse() {
    ShadowBuild.setModel("pixel 8")
    assertFalse(isAICoreSupported(null))
  }

  @Test
  fun isAICoreSupported_emptyAllowedDeviceModels_returnsFalse() {
    ShadowBuild.setModel("pixel 8")
    assertFalse(isAICoreSupported(emptySet()))
  }

  @Test
  fun isAICoreSupported_nullBuildModel_returnsFalse() {
    ShadowBuild.setModel(null)
    assertFalse(isAICoreSupported(setOf("pixel 8", "pixel 9")))
  }

  @Test
  fun isAICoreSupported_matchingDeviceModel_returnsTrue() {
    ShadowBuild.setModel("Pixel 8")
    assertTrue(isAICoreSupported(setOf("pixel 8", "pixel 9")))
  }

  @Test
  fun isAICoreSupported_nonMatchingDeviceModel_returnsFalse() {
    ShadowBuild.setModel("Pixel 7")
    assertFalse(isAICoreSupported(setOf("pixel 8", "pixel 9")))
  }

  @Test
  fun isAICoreSupported_uppercaseAndMixedCase_returnsTrue() {
    ShadowBuild.setModel("PIXEL 9 PRO")
    assertTrue(isAICoreSupported(setOf("pixel 9 pro")))
  }
}
