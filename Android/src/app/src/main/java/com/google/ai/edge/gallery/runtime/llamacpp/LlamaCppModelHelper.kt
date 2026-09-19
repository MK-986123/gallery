/*
 * Copyright 2026
 *
 * Licensed under the Apache License, Version 2.0.
 */
package com.google.ai.edge.gallery.runtime.llamacpp

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.arm.aichat.AiChat
import com.arm.aichat.InferenceEngine
import com.arm.aichat.isModelLoaded
import com.google.ai.edge.gallery.data.ConfigKeys
import com.google.ai.edge.gallery.data.DEFAULT_MAX_TOKEN
import com.google.ai.edge.gallery.data.Model
import com.google.ai.edge.gallery.data.markInitializationFailed
import com.google.ai.edge.gallery.data.markInitializationStarted
import com.google.ai.edge.gallery.data.markInitialized
import com.google.ai.edge.gallery.data.resetInitialization
import com.google.ai.edge.gallery.runtime.CleanUpListener
import com.google.ai.edge.gallery.runtime.LlmModelHelper
import com.google.ai.edge.gallery.runtime.ResultListener
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.ToolProvider
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking

private const val TAG = "AGLlamaCppModelHelper"

private data class LlamaCppModelInstance(
  val engine: InferenceEngine,
  val scope: CoroutineScope,
  var generationJob: Job? = null,
  var cleanUpListener: CleanUpListener? = null,
)

object LlamaCppModelHelper : LlmModelHelper {
  override fun initialize(
    context: Context,
    model: Model,
    taskId: String,
    supportImage: Boolean,
    supportAudio: Boolean,
    onDone: (String) -> Unit,
    systemInstruction: Contents?,
    tools: List<ToolProvider>,
    enableConversationConstrainedDecoding: Boolean,
    coroutineScope: CoroutineScope?,
  ) {
    if (model.instance != null) {
      model.markInitialized()
      onDone("")
      return
    }

    model.markInitializationStarted()
    val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    scope.launch {
      try {
        val engine = AiChat.getInferenceEngine(context)
        val state =
          engine.state.first {
            it is InferenceEngine.State.Initialized ||
              it is InferenceEngine.State.ModelReady ||
              it is InferenceEngine.State.Error
          }

        if (state is InferenceEngine.State.ModelReady || state is InferenceEngine.State.Error) {
          engine.cleanUp()
        }

        engine.loadModel(model.getPath(context))
        engine.setSystemPrompt(systemInstruction?.toString().orEmpty())

        model.markInitialized(LlamaCppModelInstance(engine = engine, scope = scope))
        onDone("")
      } catch (e: Exception) {
        Log.e(TAG, "Failed to initialize llama.cpp model '${model.name}'", e)
        model.markInitializationFailed(e)
        scope.cancel()
        onDone(e.message ?: "Failed to initialize llama.cpp model")
      }
    }
  }

  override fun resetConversation(
    model: Model,
    supportImage: Boolean,
    supportAudio: Boolean,
    systemInstruction: Contents?,
    tools: List<ToolProvider>,
    enableConversationConstrainedDecoding: Boolean,
    initialMessages: List<Message>,
  ) {
    val instance = model.instance as? LlamaCppModelInstance ?: return
    runBlocking(Dispatchers.IO) {
      instance.generationJob?.cancelAndJoin()
      instance.generationJob = null
      instance.engine.setSystemPrompt(systemInstruction?.toString().orEmpty())
    }
    if (initialMessages.isNotEmpty()) {
      Log.w(TAG, "llama.cpp session restore currently resets native context; UI history is preserved.")
    }
  }

  override fun cleanUp(model: Model, onDone: () -> Unit) {
    val instance = model.instance as? LlamaCppModelInstance
    if (instance == null) {
      model.resetInitialization()
      onDone()
      return
    }

    runBlocking(Dispatchers.IO) {
      instance.generationJob?.cancelAndJoin()
      instance.generationJob = null
      runCatching {
        val state = instance.engine.state.value
        if (state.isModelLoaded || state is InferenceEngine.State.Error) {
          instance.engine.cleanUp()
        }
      }.onFailure { Log.w(TAG, "Failed to clean up llama.cpp model", it) }
    }

    instance.cleanUpListener?.invoke()
    instance.scope.cancel()
    model.resetInitialization()
    onDone()
  }

  override fun runInference(
    model: Model,
    input: String,
    resultListener: ResultListener,
    cleanUpListener: CleanUpListener,
    onError: (message: String) -> Unit,
    images: List<Bitmap>,
    audioClips: List<ByteArray>,
    coroutineScope: CoroutineScope?,
    extraContext: Map<String, String>?,
    sessionId: String?,
    messageIndex: Int?,
  ) {
    val instance = model.instance as? LlamaCppModelInstance
    if (instance == null) {
      onError("llama.cpp model is not initialized.")
      return
    }
    if (images.isNotEmpty() || audioClips.isNotEmpty()) {
      onError("The llama.cpp GGUF backend currently supports text input only.")
      return
    }
    if (input.isBlank()) {
      onError("Prompt cannot be empty.")
      return
    }

    instance.cleanUpListener = cleanUpListener
    instance.generationJob?.cancel()
    val maxTokens =
      model.getIntConfigValue(key = ConfigKeys.MAX_TOKENS, defaultValue = DEFAULT_MAX_TOKEN)

    instance.generationJob =
      instance.scope.launch {
        try {
          instance.engine.sendUserPrompt(input, maxTokens).collect { token ->
            resultListener(token, false, null)
          }
          resultListener("", true, null)
        } catch (e: CancellationException) {
          resultListener("", true, null)
        } catch (e: Exception) {
          Log.e(TAG, "llama.cpp inference failed", e)
          onError("Error: ${e.message ?: "llama.cpp inference failed"}")
        }
      }
  }

  override fun stopResponse(model: Model) {
    val instance = model.instance as? LlamaCppModelInstance ?: return
    instance.generationJob?.cancel()
  }
}
