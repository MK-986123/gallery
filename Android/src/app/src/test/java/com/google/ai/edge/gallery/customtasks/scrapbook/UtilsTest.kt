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

  private companion object {
    const val CONTRAST_DELTA = 0.01f
    const val LUMINANCE_DELTA = 0.001f
  }

  private data class ContrastTestCase(
    val color1: Color,
    val color2: Color,
    val expectedRatio: Float,
    val name: String,
  )

  @Test
  fun testCalculateContrastRatio() {
    val testCases = listOf(
      // Black and White: L1 = 1.0, L2 = 0.0 -> (1.05) / (0.05) = 21:1
      ContrastTestCase(Color.Black, Color.White, 21.0f, "Black vs White"),
      ContrastTestCase(Color.White, Color.Black, 21.0f, "White vs Black (Order independence)"),
      // Identical colors: L1 == L2 -> (L + 0.05) / (L + 0.05) = 1.0
      ContrastTestCase(Color.Black, Color.Black, 1.0f, "Black vs Black"),
      ContrastTestCase(Color.White, Color.White, 1.0f, "White vs White"),
      ContrastTestCase(Color.Red, Color.Red, 1.0f, "Red vs Red"),
      ContrastTestCase(Color.Blue, Color.Blue, 1.0f, "Blue vs Blue"),
      // Known contrast ratios
      // Red (0.2126) vs White (1.0): (1.05) / (0.2126 + 0.05) = 1.05 / 0.2626 ~ 3.998f
      ContrastTestCase(Color.Red, Color.White, 4.00f, "Red vs White"),
      // Blue (0.0722) vs White (1.0): (1.05) / (0.0722 + 0.05) = 1.05 / 0.1222 ~ 8.59f
      ContrastTestCase(Color.Blue, Color.White, 8.59f, "Blue vs White"),
      // Green (0.7152) vs Black (0.0): (0.7152 + 0.05) / (0.05) = 0.7652 / 0.05 ~ 15.30f
      ContrastTestCase(Color.Green, Color.Black, 15.30f, "Green vs Black"),
      // Dark gray (RGB 0.1, 0.1, 0.1) vs Light gray (RGB 0.9, 0.9, 0.9)
      ContrastTestCase(
        Color(0.1f, 0.1f, 0.1f, 1.0f),
        Color(0.9f, 0.9f, 0.9f, 1.0f),
        14.94f,
        "Dark Gray vs Light Gray",
      ),
    )

    for (testCase in testCases) {
      val actualRatio = calculateContrastRatio(testCase.color1, testCase.color2)
      assertEquals(
        "Failed test case: ${testCase.name}",
        testCase.expectedRatio,
        actualRatio,
        CONTRAST_DELTA,
      )
    }
  }

  @Test
  fun relativeLuminance_black_returnsZero() {
    assertEquals(0.0f, Color.Black.relativeLuminance(), LUMINANCE_DELTA)
  }

  @Test
  fun relativeLuminance_white_returnsOne() {
    assertEquals(1.0f, Color.White.relativeLuminance(), LUMINANCE_DELTA)
  }

  @Test
  fun relativeLuminance_pureRed_returnsRedWeight() {
    // Standard sRGB weight for red is 0.2126
    assertEquals(0.2126f, Color.Red.relativeLuminance(), LUMINANCE_DELTA)
  }

  @Test
  fun relativeLuminance_pureGreen_returnsGreenWeight() {
    // Standard sRGB weight for green is 0.7152
    assertEquals(0.7152f, Color.Green.relativeLuminance(), LUMINANCE_DELTA)
  }

  @Test
  fun relativeLuminance_pureBlue_returnsBlueWeight() {
    // Standard sRGB weight for blue is 0.0722
    assertEquals(0.0722f, Color.Blue.relativeLuminance(), LUMINANCE_DELTA)
  }

  @Test
  fun relativeLuminance_lowValueComponents_executesLinearBranch() {
    // Tests color component <= 0.03928f branch in convertToLinear
    val lowColor = Color(0.03f, 0.02f, 0.01f, 1.0f)
    val expectedLinearR = 0.03f / 12.92f
    val expectedLinearG = 0.02f / 12.92f
    val expectedLinearB = 0.01f / 12.92f
    val expectedLuminance =
      0.2126f * expectedLinearR + 0.7152f * expectedLinearG + 0.0722f * expectedLinearB

    assertEquals(expectedLuminance, lowColor.relativeLuminance(), LUMINANCE_DELTA)
  }
}
