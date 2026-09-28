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

package com.google.ai.edge.gallery.common

import com.google.ai.edge.gallery.data.CategoryInfo
import com.google.ai.edge.gallery.data.SystemPromptRepository
import com.google.ai.edge.gallery.data.Task
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flowOf
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Test

class SystemPromptHelperTest {

  private val sampleTask = Task(
    id = "test_task",
    label = "Test Task",
    category = CategoryInfo("test_cat", label = "Test Category"),
    description = "Test Description",
    models = mutableListOf(),
    defaultSystemPrompt = "Default System Prompt"
  )

  @Test
  fun testGetEffectiveSystemPrompt_nullRepository_returnsDefaultPrompt() = runBlocking {
    val effectivePrompt = SystemPromptHelper.getEffectiveSystemPrompt(repo = null, task = sampleTask)
    assertEquals("Default System Prompt", effectivePrompt)
  }

  private class FakeSystemPromptRepository(
    private val customPromptMap: Map<String, String?>
  ) : SystemPromptRepository(FakeUserDataStore()) {
    override fun getCustomSystemPrompt(taskId: String): Flow<String?> {
      return flowOf(customPromptMap[taskId])
    }
  }

  private class FakeUserDataStore : androidx.datastore.core.DataStore<com.google.ai.edge.gallery.proto.UserData> {
    override val data: Flow<com.google.ai.edge.gallery.proto.UserData>
      get() = flowOf(com.google.ai.edge.gallery.proto.UserData.getDefaultInstance())
    override suspend fun updateData(
      transform: suspend (t: com.google.ai.edge.gallery.proto.UserData) -> com.google.ai.edge.gallery.proto.UserData
    ): com.google.ai.edge.gallery.proto.UserData {
      return com.google.ai.edge.gallery.proto.UserData.getDefaultInstance()
    }
  }

  @Test
  fun testGetEffectiveSystemPrompt_customPromptExists_returnsCustomPrompt() = runBlocking {
    val repo = FakeSystemPromptRepository(mapOf("test_task" to "Custom System Prompt"))
    val effectivePrompt = SystemPromptHelper.getEffectiveSystemPrompt(repo = repo, task = sampleTask)
    assertEquals("Custom System Prompt", effectivePrompt)
  }

  @Test
  fun testGetEffectiveSystemPrompt_customPromptNull_returnsDefaultPrompt() = runBlocking {
    val repo = FakeSystemPromptRepository(mapOf("test_task" to null))
    val effectivePrompt = SystemPromptHelper.getEffectiveSystemPrompt(repo = repo, task = sampleTask)
    assertEquals("Default System Prompt", effectivePrompt)
  }

  @Test
  fun testGetEffectiveSystemPrompt_customPromptEmptyString_returnsCustomEmptyPrompt() = runBlocking {
    val repo = FakeSystemPromptRepository(mapOf("test_task" to ""))
    val effectivePrompt = SystemPromptHelper.getEffectiveSystemPrompt(repo = repo, task = sampleTask)
    assertEquals("", effectivePrompt)
  }
}
