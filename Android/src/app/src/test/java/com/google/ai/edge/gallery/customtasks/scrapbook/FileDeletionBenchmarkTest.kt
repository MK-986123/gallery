package com.google.ai.edge.gallery.customtasks.scrapbook

import java.io.File
import java.nio.file.Files
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.async
import kotlinx.coroutines.awaitAll
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withContext
import org.junit.Assert.assertFalse
import org.junit.Test

class FileDeletionBenchmarkTest {

  private suspend fun deleteFilesSequentialInLoop(cutoutFiles: List<Pair<File, File>>) {
    withContext(Dispatchers.IO) {
      for ((origFile, currFile) in cutoutFiles) {
        deleteFilesSequential(listOf(origFile, currFile))
      }
    }
  }

  private suspend fun deleteFilesSequential(files: List<File>) {
    withContext(Dispatchers.IO) {
      for (file in files) {
        try {
          if (file.exists()) {
            file.delete()
          }
        } catch (e: Exception) {
          // ignore
        }
      }
    }
  }

  private suspend fun deleteFilesBatchParallel(files: List<File>) {
    withContext(Dispatchers.IO) {
      files.map { file ->
        async {
          try {
            if (file.exists()) {
              file.delete()
            }
          } catch (e: Exception) {
            // ignore
          }
        }
      }.awaitAll()
    }
  }

  @Test
  fun testFileDeletionCorrectnessAndBenchmark() {
    runBlocking {
      val tempDir = Files.createTempDirectory("file_deletion_test").toFile()
      tempDir.deleteOnExit()

      val numCutouts = 100

      // Measure loop sequential
      val cutoutFilesSequential = (1..numCutouts).map { id ->
        val orig = File(tempDir, "seq_cutout_${id}_original.png").apply { writeText("sample data $id") }
        val curr = File(tempDir, "seq_cutout_${id}_current.png").apply { writeText("sample data $id") }
        Pair(orig, curr)
      }

      val startSeq = System.currentTimeMillis()
      deleteFilesSequentialInLoop(cutoutFilesSequential)
      val endSeq = System.currentTimeMillis()
      val seqDuration = endSeq - startSeq

      for ((orig, curr) in cutoutFilesSequential) {
        assertFalse("File should be deleted: ${orig.name}", orig.exists())
        assertFalse("File should be deleted: ${curr.name}", curr.exists())
      }

      // Measure batch parallel
      val cutoutFilesParallel = (1..numCutouts).map { id ->
        val orig = File(tempDir, "par_cutout_${id}_original.png").apply { writeText("sample data $id") }
        val curr = File(tempDir, "par_cutout_${id}_current.png").apply { writeText("sample data $id") }
        Pair(orig, curr)
      }

      val allParallelFiles = cutoutFilesParallel.flatMap { listOf(it.first, it.second) }

      val startPar = System.currentTimeMillis()
      deleteFilesBatchParallel(allParallelFiles)
      val endPar = System.currentTimeMillis()
      val parDuration = endPar - startPar

      for (file in allParallelFiles) {
        assertFalse("File should be deleted: ${file.name}", file.exists())
      }

      println("=== BENCHMARK RESULTS FOR 100 CUTOUTS (200 FILES) ===")
      println("Sequential file deletion in loop: ${seqDuration} ms")
      println("Batch parallel file deletion:     ${parDuration} ms")
      println("====================================================")

      tempDir.deleteRecursively()
    }
  }
}
