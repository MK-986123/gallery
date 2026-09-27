package com.google.ai.edge.gallery.ui.modelmanager

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

class AllowlistVersionTest {
  @Test
  fun previousAllowlistVersionDecrementsPatch() {
    assertEquals("1_0_19", getPreviousAllowlistVersion("1_0_20"))
    assertEquals("1_0_9", getPreviousAllowlistVersion("1_0_10"))
    assertEquals("1_1_0", getPreviousAllowlistVersion("1_1_1"))
    assertNull(getPreviousAllowlistVersion("1_1_0"))
    assertNull(getPreviousAllowlistVersion("1_0_20-dev"))
  }
}
