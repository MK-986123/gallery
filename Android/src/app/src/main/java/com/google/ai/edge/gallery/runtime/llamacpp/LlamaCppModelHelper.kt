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
import com.arm.aichat.ChatRole
import com.arm.aichat.ChatTurn
import com.arm.aichat.InferenceEngine
import com.arm.aichat.SamplingSettings
import com.arm.aichat.isModelLoaded
import com.google.ai.edge.gallery.data.ConfigKeys
import com.google.ai.edge.gallery.data.DEFAULT_MAX_TOKEN
import com.google.ai.edge.gallery.data.DEFAULT_TOPK
import com.google.ai.edge.gallery.data.DEFAULT_TOPP
import com.google.ai.edge.gallery.data.DEFAULT_TEMPERATURE
import com.google.ai.edge.gallery.data.Model
import com.google.ai.edge.gallery.data.markInitializationFailed
import com.google.ai.edge.gallery.data.markInitializationStarted
import com.google.ai.edge.gallery.data.markInitialized
import com.google.ai.edge.gallery.data.resetInitialization
import com.google.ai.edge.gallery.huggingface.supportsGgufOnDevice
import com.google.ai.edge.gallery.runtime.CleanUpListener
import com.google.ai.edge.gallery.runtime.LlmModelHelper
import com.google.ai.edge.gallery.runtime.ResultListener
import com.google.ai.edge.litertlm.Contents
import com.google.ai.edge.litertlm.Message
import com.google.ai.edge.litertlm.Role
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
  var systemPrompt: String = "",
  var turns: List<ChatTurn> = emptyList(),
  @Volatile
  var omittedRestoreTurns: Int = 0,
  var restoreFailed: Boolean = false,
)

object LlamaCppModelHelper : LlmModelHelper {
  fun takeOmittedRestoreTurns(model: Model): Int {
    val instance = model.instance as? LlamaCppModelInstance ?: return 0
    return instance.omittedRestoreTurns.also { instance.omittedRestoreTurns = 0 }
  }

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
    if (!supportsGgufOnDevice()) {
      onDone("GGUF inference requires an arm64 Android device.")
      return
    }
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

        if (state is InferenceEngine.State.Error || state is InferenceEngine.State.ModelReady) {
          engine.cleanUp()
        }

        engine.loadModel(model.getPath(context))
        engine.setSystemPrompt(systemInstruction?.toString().orEmpty())

        model.markInitialized(
          LlamaCppModelInstance(
            engine = engine, scope = scope,
            systemPrompt = systemInstruction?.toString().orEmpty(),
          )
        )
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
      try {
        val turns = initialMessages.map { message ->
          ChatTurn(
            role = if (message.role == Role.USER) ChatRole.USER else ChatRole.ASSISTANT,
            text = message.contents.toString(),
          )
        }
        val reserveTokens =
          model.getIntConfigValue(key = ConfigKeys.MAX_TOKENS, defaultValue = DEFAULT_MAX_TOKEN)
        instance.omittedRestoreTurns =
          instance.engine.restoreConversation(
            systemInstruction?.toString().orEmpty(), turns, reserveTokens
          )
        instance.systemPrompt = systemInstruction?.toString().orEmpty()
        instance.turns = turns.drop(instance.omittedRestoreTurns)
        instance.restoreFailed = false
      } catch (e: Exception) {
        instance.restoreFailed = true
        throw e
      }
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
    if (instance.restoreFailed) {
      onError("Saved GGUF chat context could not be restored. Retry the chat or start a new one.")
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
    val sampling =
      SamplingSettings(
        topK = model.getIntConfigValue(ConfigKeys.TOPK, DEFAULT_TOPK),
        topP = model.getFloatConfigValue(ConfigKeys.TOPP, DEFAULT_TOPP),
        temperature = model.getFloatConfigValue(ConfigKeys.TEMPERATURE, DEFAULT_TEMPERATURE),
      )

    instance.generationJob =
      instance.scope.launch {
        try {
          // Rebuild from complete saved turns before each request. This also recovers cleanly
          // after a cancelled or partially generated native response.
          instance.omittedRestoreTurns =
            instance.engine.restoreConversation(instance.systemPrompt, instance.turns, maxTokens)
          instance.turns = instance.turns.drop(instance.omittedRestoreTurns)
          instance.engine.setSampling(sampling)
          val response = StringBuilder()
          instance.engine.sendUserPrompt(input, maxTokens).collect { token ->
            response.append(token)
            resultListener(token, false, null)
          }
          instance.turns = instance.turns + ChatTurn(ChatRole.USER, input) +
            ChatTurn(ChatRole.ASSISTANT, response.toString())
          resultListener("", true, null)
        } catch (e: CancellationException) {
          resultListener("", true, null)
        } catch (e: Exception) {
          instance.restoreFailed = true
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
