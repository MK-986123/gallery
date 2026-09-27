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
  fun processLlmResponse_replacesLiteralNewlinesWithActualNewlines() {
    val input = "Hello\\nWorld"
    val expected = "Hello\nWorld"
    assertEquals(expected, processLlmResponse(input))
  }

  @Test
  fun processLlmResponse_handlesMultipleLiteralNewlines() {
    val input = "Line 1\\nLine 2\\nLine 3"
    val expected = "Line 1\nLine 2\nLine 3"
    assertEquals(expected, processLlmResponse(input))
  }

  @Test
  fun processLlmResponse_preservesActualNewlines() {
    val input = "Hello\nWorld"
    val expected = "Hello\nWorld"
    assertEquals(expected, processLlmResponse(input))
  }

  @Test
  fun processLlmResponse_handlesMixedNewlines() {
    val input = "Line 1\nLine 2\\nLine 3"
    val expected = "Line 1\nLine 2\nLine 3"
    assertEquals(expected, processLlmResponse(input))
  }

  @Test
  fun processLlmResponse_handlesEmptyString() {
    assertEquals("", processLlmResponse(""))
  }

  @Test
  fun processLlmResponse_handlesStringWithoutNewlines() {
    val input = "Hello World"
    assertEquals(input, processLlmResponse(input))
  }

  @Test
  fun processLlmResponse_handlesLeadingAndTrailingLiteralNewlines() {
    val input = "\\nHello World\\n"
    val expected = "\nHello World\n"
    assertEquals(expected, processLlmResponse(input))
  }
}
