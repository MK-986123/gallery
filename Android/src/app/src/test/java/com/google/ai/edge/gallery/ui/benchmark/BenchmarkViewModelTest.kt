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

package com.google.ai.edge.gallery.ui.benchmark

import android.content.Context
import android.content.ContextWrapper
import com.google.ai.edge.gallery.data.DataStoreRepository
import com.google.ai.edge.gallery.proto.AccessTokenData
import com.google.ai.edge.gallery.proto.BenchmarkResult
import com.google.ai.edge.gallery.proto.Cutout
import com.google.ai.edge.gallery.proto.ImportedModel
import com.google.ai.edge.gallery.proto.Skill
import com.google.ai.edge.gallery.proto.Theme
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class BenchmarkViewModelTest {

  private val fakeRepo = FakeDataStoreRepository()
  private val fakeContext = ContextWrapper(null)

  @Test
  fun testSetErrorMessage() {
    val viewModel = BenchmarkViewModel(appContext = fakeContext, dataStoreRepository = fakeRepo)
    assertNull(viewModel.uiState.value.errorMessage)

    viewModel.setErrorMessage("Test Error")
    assertEquals("Test Error", viewModel.uiState.value.errorMessage)

    viewModel.setErrorMessage(null)
    assertNull(viewModel.uiState.value.errorMessage)
  }
}

private class FakeDataStoreRepository : DataStoreRepository {
  private val results = mutableListOf<BenchmarkResult>()

  override fun saveTextInputHistory(history: List<String>) {}
  override fun readTextInputHistory(): List<String> = emptyList()
  override fun saveTheme(theme: Theme) {}
  override fun readTheme(): Theme = Theme.THEME_AUTO
  override fun saveFirebaseAnalytics(enabled: Boolean) {}
  override fun readFirebaseAnalytics(): Boolean = true
  override fun saveSecret(key: String, value: String) {}
  override fun readSecret(key: String): String? = null
  override fun deleteSecret(key: String) {}
  override fun saveAccessTokenData(accessToken: String, refreshToken: String, expiresAt: Long) {}
  override fun clearAccessTokenData() {}
  override fun readAccessTokenData(): AccessTokenData? = null
  override fun saveImportedModels(importedModels: List<ImportedModel>) {}
  override fun readImportedModels(): List<ImportedModel> = emptyList()
  override fun isTosAccepted(): Boolean = true
  override fun acceptTos() {}
  override fun isGemmaTermsOfUseAccepted(): Boolean = true
  override fun acceptGemmaTermsOfUse() {}
  override fun getHasRunTinyGarden(): Boolean = false
  override fun setHasRunTinyGarden(hasRun: Boolean) {}
  override fun addCutout(cutout: Cutout) {}
  override fun getAllCutouts(): List<Cutout> = emptyList()
  override fun setCutout(newCutout: Cutout) {}
  override fun setCutouts(cutouts: List<Cutout>) {}
  override fun setHasSeenBenchmarkComparisonHelp(seen: Boolean) {}
  override fun getHasSeenBenchmarkComparisonHelp(): Boolean = false
  override fun addBenchmarkResult(result: BenchmarkResult) { results.add(result) }
  override fun getAllBenchmarkResults(): List<BenchmarkResult> = results
  override fun deleteBenchmarkResult(index: Int) { if (index in results.indices) results.removeAt(index) }
  override fun addSkill(skill: Skill) {}
  override fun setSkills(skills: List<Skill>) {}
  override fun setSkillSelected(skill: Skill, selected: Boolean) {}
  override fun setAllSkillsSelected(selected: Boolean) {}
  override fun getAllSkills(): List<Skill> = emptyList()
  override fun deleteSkill(name: String) {}
  override suspend fun deleteSkills(names: Set<String>) {}
  override fun addViewedPromoId(promoId: String) {}
  override fun removeViewedPromoId(promoId: String) {}
  override fun hasViewedPromo(promoId: String): Boolean = false
}
