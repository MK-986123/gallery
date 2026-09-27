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

import org.junit.Assert.assertEquals
import org.junit.Test

class UtilsTest {

  @Test
  fun cleanUpMediapipeTaskErrorMessage_withSourceLocationTrace_removesTraceAndFollowingText() {
    val input =
      "Failed to run task === Source Location Trace:\n  at mediapipe/tasks/cc/vision/image_segmenter/image_segmenter.cc:123"
    val expected = "Failed to run task "
    assertEquals(expected, cleanUpMediapipeTaskErrorMessage(input))
  }

  @Test
  fun cleanUpMediapipeTaskErrorMessage_withoutSourceLocationTrace_returnsOriginalMessage() {
    val input = "Failed to run task due to missing model file"
    assertEquals(input, cleanUpMediapipeTaskErrorMessage(input))
  }

  @Test
  fun cleanUpMediapipeTaskErrorMessage_emptyString_returnsEmptyString() {
    val input = ""
    assertEquals("", cleanUpMediapipeTaskErrorMessage(input))
  }

  @Test
  fun cleanUpMediapipeTaskErrorMessage_startsWithSourceLocationTrace_returnsEmptyString() {
    val input = "=== Source Location Trace:\n  at mediapipe/tasks/cc/vision/image_segmenter/image_segmenter.cc:123"
    assertEquals("", cleanUpMediapipeTaskErrorMessage(input))
  }

  @Test
  fun cleanUpMediapipeTaskErrorMessage_multipleTraces_truncatesAtFirstTrace() {
    val input = "Error msg === Source Location Trace: first === Source Location Trace: second"
    val expected = "Error msg "
    assertEquals(expected, cleanUpMediapipeTaskErrorMessage(input))
  }
}
