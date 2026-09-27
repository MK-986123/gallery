/*
 * Copyright 2026 Google LLC
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

package com.google.ai.edge.gallery.customtasks.scrapbook

import androidx.compose.ui.graphics.Color
import org.junit.Assert.assertEquals
import org.junit.Test

class UtilsTest {

  private val delta = 0.0001f

  @Test
  fun relativeLuminance_black_returnsZero() {
    val luminance = Color.Black.relativeLuminance()
    assertEquals(0.0f, luminance, delta)
  }

  @Test
  fun relativeLuminance_white_returnsOne() {
    val luminance = Color.White.relativeLuminance()
    assertEquals(1.0f, luminance, delta)
  }

  @Test
  fun relativeLuminance_pureRed_returnsRedWeight() {
    val luminance = Color.Red.relativeLuminance()
    assertEquals(0.2126f, luminance, delta)
  }

  @Test
  fun relativeLuminance_pureGreen_returnsGreenWeight() {
    val luminance = Color.Green.relativeLuminance()
    assertEquals(0.7152f, luminance, delta)
  }

  @Test
  fun relativeLuminance_pureBlue_returnsBlueWeight() {
    val luminance = Color.Blue.relativeLuminance()
    assertEquals(0.0722f, luminance, delta)
  }

  @Test
  fun relativeLuminance_lowComponentValue_usesLowLinearConversionBranch() {
    // Component value <= 0.03928f converts via component / 12.92f
    val colorComponent = 0.03f
    val color = Color(colorComponent, colorComponent, colorComponent)
    // Compose Color quantizes color components, so use color.red/green/blue
    val r = color.red / 12.92f
    val g = color.green / 12.92f
    val b = color.blue / 12.92f
    val expectedLuminance = 0.2126f * r + 0.7152f * g + 0.0722f * b
    val luminance = color.relativeLuminance()
    assertEquals(expectedLuminance, luminance, delta)
  }

  @Test
  fun relativeLuminance_highComponentValue_usesHighLinearConversionBranch() {
    // Component value > 0.03928f converts via ((component + 0.055f) / 1.055f)^3
    val colorComponent = 0.5f
    val color = Color(colorComponent, colorComponent, colorComponent)
    val r = ((color.red + 0.055f) / 1.055f).let { it * it * it }
    val g = ((color.green + 0.055f) / 1.055f).let { it * it * it }
    val b = ((color.blue + 0.055f) / 1.055f).let { it * it * it }
    val expectedLuminance = 0.2126f * r + 0.7152f * g + 0.0722f * b
    val luminance = color.relativeLuminance()
    assertEquals(expectedLuminance, luminance, delta)
  }

  @Test
  fun calculateContrastRatio_blackAndWhite_returnsMaxContrast() {
    val ratio = calculateContrastRatio(Color.Black, Color.White)
    // Formula: (1.0 + 0.05) / (0.0 + 0.05) = 1.05 / 0.05 = 21.0
    assertEquals(21.0f, ratio, delta)
  }

  @Test
  fun calculateContrastRatio_sameColor_returnsMinContrast() {
    val ratio = calculateContrastRatio(Color.White, Color.White)
    assertEquals(1.0f, ratio, delta)

    val blackRatio = calculateContrastRatio(Color.Black, Color.Black)
    assertEquals(1.0f, blackRatio, delta)
  }

  @Test
  fun calculateContrastRatio_isSymmetric() {
    val ratio1 = calculateContrastRatio(Color.Red, Color.Blue)
    val ratio2 = calculateContrastRatio(Color.Blue, Color.Red)
    assertEquals(ratio1, ratio2, delta)
  }
}
