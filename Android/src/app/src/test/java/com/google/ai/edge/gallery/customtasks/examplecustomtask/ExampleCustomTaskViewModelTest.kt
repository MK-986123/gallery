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

package com.google.ai.edge.gallery.customtasks.examplecustomtask

import androidx.compose.ui.graphics.Color
import org.junit.Assert.assertEquals
import org.junit.Test

class ExampleCustomTaskViewModelTest {

  @Test
  fun defaultState_textColorIsBlack() {
    val viewModel = ExampleCustomTaskViewModel()
    assertEquals(Color.Black, viewModel.uiState.value.textColor)
  }

  @Test
  fun updateTextColor_updatesUiStateWithNewColor() {
    val viewModel = ExampleCustomTaskViewModel()

    viewModel.updateTextColor(Color.Red)

    assertEquals(Color.Red, viewModel.uiState.value.textColor)
  }

  @Test
  fun updateTextColor_multipleUpdates_updatesUiStateSequentially() {
    val viewModel = ExampleCustomTaskViewModel()

    viewModel.updateTextColor(Color.Red)
    assertEquals(Color.Red, viewModel.uiState.value.textColor)

    viewModel.updateTextColor(Color.Blue)
    assertEquals(Color.Blue, viewModel.uiState.value.textColor)

    viewModel.updateTextColor(Color.Green)
    assertEquals(Color.Green, viewModel.uiState.value.textColor)
  }
}
