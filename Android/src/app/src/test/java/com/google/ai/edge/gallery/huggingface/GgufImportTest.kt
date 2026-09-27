package com.google.ai.edge.gallery.huggingface

import com.google.ai.edge.gallery.data.Accelerator
import com.google.ai.edge.gallery.data.BackendSpec
import com.google.ai.edge.gallery.data.RuntimeType
import java.io.ByteArrayInputStream
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class GgufImportTest {
  @Test
  fun parsesHuggingFaceGgufFileAndModelCard() {
    val file =
      HfUrlInfo.parse("https://huggingface.co/example/bonsai/resolve/main/Bonsai-Q1_0.GGUF?download=true")
    assertEquals("example/bonsai", file.modelId)
    assertEquals("Bonsai-Q1_0.GGUF", file.fileName)
    assertTrue(file.isDirectModelFile)

    val card = HfUrlInfo.parse("https://huggingface.co/example/bonsai")
    assertEquals("example/bonsai", card.modelId)
    assertFalse(card.isDirectModelFile)
    assertTrue(isSupportedModelFileName("Bonsai-Q1_0.gguf"))
    assertTrue(isSupportedModelFileName("model.litertlm"))
    assertEquals(
      "https://huggingface.co/example/bonsai/resolve/v1/subdir/model.gguf?download=true",
      normalizeDirectModelFileUrl(
        "https://huggingface.co/example/bonsai/blob/v1/subdir/model.gguf?download=true"
      ),
    )
    assertEquals(
      "https://models.example.org/blob/main/model.gguf",
      normalizeDirectModelFileUrl("https://models.example.org/blob/main/model.gguf"),
    )
    assertEquals(
      "https://huggingface.co/example/bonsai/resolve/main/subdir/model.gguf",
      normalizeDirectModelFileUrl("example/bonsai/subdir/model.gguf"),
    )
  }

  @Test
  fun distinguishesGgufHeaderAndDeviceAbi() {
    assertTrue(hasGgufHeader(ByteArrayInputStream("GGUFpayload".toByteArray())))
    val chunked =
      object : ByteArrayInputStream("GGUFpayload".toByteArray()) {
        override fun read(buffer: ByteArray, offset: Int, length: Int): Int =
          super.read(buffer, offset, minOf(length, 1))
      }
    assertTrue(hasGgufHeader(chunked))
    assertFalse(hasGgufHeader(ByteArrayInputStream("not-a-gguf".toByteArray())))
    assertFalse(hasGgufHeader(ByteArrayInputStream(byteArrayOf(0x47, 0x47))))

    val supported = DeviceHardwareInfo(DeviceVendor.GENERIC_ANDROID, "", true)
    val unsupported = DeviceHardwareInfo(DeviceVendor.GENERIC_ANDROID, "", false)
    assertTrue(supported.isCompatibleWithFile("Bonsai-Q1_0.gguf"))
    assertFalse(unsupported.isCompatibleWithFile("Bonsai-Q1_0.gguf"))
    assertTrue(unsupported.isCompatibleWithFile("model.litertlm"))
  }

  @Test
  fun keepsGgufAndLiteRtRuntimeMappingsSeparate() {
    val gguf = BackendSpec(RuntimeType.LLAMA_CPP, accelerators = listOf(Accelerator.CPU))
    val liteRt = BackendSpec(RuntimeType.LITERT_LM, accelerators = listOf(Accelerator.GPU))
    assertTrue(gguf.isLlamaCpp)
    assertFalse(gguf.isLiteRtLm)
    assertEquals(Accelerator.CPU, gguf.defaultAccelerator)
    assertTrue(liteRt.isLiteRtLm)
    assertFalse(liteRt.isLlamaCpp)
  }
}
