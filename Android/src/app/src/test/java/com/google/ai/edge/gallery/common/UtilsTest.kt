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

import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class UtilsTest {

  @Test
  fun convertStringToJsonObject_validJsonObject_returnsParsedObject() {
    val jsonString = """{"name": "Gallery", "version": 1, "enabled": true}"""
    val result = convertStringToJsonObject(jsonString)

    assertEquals(3, result.size)
    assertEquals("Gallery", result["name"]?.jsonPrimitive?.content)
    assertEquals("1", result["version"]?.jsonPrimitive?.content)
    assertEquals("true", result["enabled"]?.jsonPrimitive?.content)
  }

  @Test
  fun convertStringToJsonObject_lenientJsonObject_returnsParsedObject() {
    val jsonString = """{name: Gallery, version: 1}"""
    val result = convertStringToJsonObject(jsonString)

    assertEquals(2, result.size)
    assertEquals("Gallery", result["name"]?.jsonPrimitive?.content)
    assertEquals("1", result["version"]?.jsonPrimitive?.content)
  }

  @Test
  fun convertStringToJsonObject_emptyJsonObject_returnsEmptyObject() {
    val jsonString = "{}"
    val result = convertStringToJsonObject(jsonString)

    assertTrue(result.isEmpty())
  }

  @Test
  fun convertStringToJsonObject_nestedJsonObject_returnsParsedNestedObject() {
    val jsonString = """{"outer": {"inner": "value"}}"""
    val result = convertStringToJsonObject(jsonString)

    assertEquals(1, result.size)
    val innerObj = result["outer"]?.jsonObject
    assertEquals("value", innerObj?.get("inner")?.jsonPrimitive?.content)
  }

  @Test
  fun convertStringToJsonObject_jsonArray_returnsEmptyObjectFallback() {
    val jsonString = """[1, 2, 3]"""
    val result = convertStringToJsonObject(jsonString)

    assertTrue(result.isEmpty())
  }

  @Test
  fun convertStringToJsonObject_jsonPrimitiveString_returnsEmptyObjectFallback() {
    val jsonString = """"just a string""""
    val result = convertStringToJsonObject(jsonString)

    assertTrue(result.isEmpty())
  }

  @Test
  fun convertStringToJsonObject_jsonPrimitiveNumber_returnsEmptyObjectFallback() {
    val jsonString = "12345"
    val result = convertStringToJsonObject(jsonString)

    assertTrue(result.isEmpty())
  }

  @Test
  fun convertStringToJsonObject_jsonPrimitiveBoolean_returnsEmptyObjectFallback() {
    val jsonString = "true"
    val result = convertStringToJsonObject(jsonString)

    assertTrue(result.isEmpty())
  }

  @Test
  fun convertStringToJsonObject_jsonPrimitiveNull_returnsEmptyObjectFallback() {
    val jsonString = "null"
    val result = convertStringToJsonObject(jsonString)

    assertTrue(result.isEmpty())
  }

  @Test
  fun convertStringToJsonObject_invalidJson_returnsEmptyObjectFallback() {
    val jsonString = "not valid json {"
    val result = convertStringToJsonObject(jsonString)

    assertTrue(result.isEmpty())
  }

  @Test
  fun convertStringToJsonObject_emptyString_returnsEmptyObjectFallback() {
    val jsonString = ""
    val result = convertStringToJsonObject(jsonString)

    assertTrue(result.isEmpty())
  }

  @Test
  fun convertStringToJsonObject_whitespaceString_returnsEmptyObjectFallback() {
    val jsonString = "   \n\t  "
    val result = convertStringToJsonObject(jsonString)

    assertTrue(result.isEmpty())
  }
}
