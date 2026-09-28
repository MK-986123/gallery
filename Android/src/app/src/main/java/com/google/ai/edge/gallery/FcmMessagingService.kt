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

package com.google.ai.edge.gallery

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.media.RingtoneManager
import android.os.Build
import android.util.Log
import androidx.core.app.NotificationCompat
import androidx.core.net.toUri
import com.google.firebase.messaging.FirebaseMessagingService
import com.google.firebase.messaging.RemoteMessage

/** Service for handling Firebase Cloud Messaging (FCM) notifications and token registration. */
class GalleryFcmMessagingService : FirebaseMessagingService() {
  override fun onMessageReceived(remoteMessage: RemoteMessage) {
    Log.d(TAG, "Full message: $remoteMessage")
    Log.d(TAG, "From: ${remoteMessage.from}")

    // Combine data and notification payloads
    val data = remoteMessage.data
    val notification = remoteMessage.notification

    val deeplink = data["deeplink"]
    val imageUrlStr = data["image_url"]
    val imageUrl = imageUrlStr?.let { it.toUri() }

    // Prefer data title/body, fallback to notification
    val title = data["title"] ?: notification?.title
    val body = data["body"] ?: notification?.body

    Log.d(TAG, "Extracted FCM Data -> Title: $title, Body: $body, Deeplink: $deeplink")

    if (title != null && body != null) {
      sendNotification(title, body, imageUrl, deeplink)
    } else if (data.isNotEmpty()) {
      handleNow()
    }
  }

  override fun onNewToken(token: String) {
    super.onNewToken(token)
    Log.d(TAG, "Refreshed FCM registration token: $token")
  }

  private fun handleNow() {
    Log.d(TAG, "Short lived task is done.")
  }

  private fun sendNotification(
    title: String?,
    messageBody: String,
    imageUrl: android.net.Uri?,
    deeplink: String? = null,
  ) {
    val intent =
      if (!deeplink.isNullOrEmpty()) {
        Intent(Intent.ACTION_VIEW, deeplink.toUri()).apply {
          flags = Intent.FLAG_ACTIVITY_NEW_TASK
        }
      } else {
        Intent(this, MainActivity::class.java).apply { addFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP) }
      }
    val requestCode = System.currentTimeMillis().toInt()
    val pendingIntent =
      PendingIntent.getActivity(
        this,
        requestCode,
        intent,
        PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT,
      )

    val defaultSoundUri = RingtoneManager.getDefaultUri(RingtoneManager.TYPE_NOTIFICATION)
    val notificationBuilder =
      NotificationCompat.Builder(this, CHANNEL_ID)
        .setSmallIcon(R.mipmap.ic_launcher)
        .setContentTitle(title ?: getString(R.string.gallery_news_notification_title))
        .setContentText(messageBody)
        .setAutoCancel(true)
        .setSound(defaultSoundUri)
        .setContentIntent(pendingIntent)
        .setPriority(NotificationCompat.PRIORITY_HIGH)

    if (imageUrl != null) {
      try {
        val url = java.net.URL(imageUrl.toString())
        val connection = url.openConnection()
        connection.connectTimeout = CONNECT_TIMEOUT_MS
        connection.readTimeout = READ_TIMEOUT_MS
        connection.getInputStream().use { inputStream ->
          val bitmap = android.graphics.BitmapFactory.decodeStream(inputStream)
          if (bitmap != null) {
            notificationBuilder.setLargeIcon(bitmap)
            notificationBuilder.setStyle(
              NotificationCompat.BigPictureStyle()
                .bigPicture(bitmap)
                .bigLargeIcon(null as android.graphics.Bitmap?)
            )
          }
        }
      } catch (e: Exception) {
        Log.w(TAG, "Failed to download image from $imageUrl", e)
      }
    }

    val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager

    // Since android Oreo notification channel is needed.
    if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
      val channel =
        NotificationChannel(
          CHANNEL_ID,
          getString(R.string.gallery_news_notification_title),
          NotificationManager.IMPORTANCE_HIGH,
        )
      notificationManager.createNotificationChannel(channel)
    }

    val notificationId = (title.hashCode() xor messageBody.hashCode() xor System.currentTimeMillis().toInt())
    notificationManager.notify(notificationId, notificationBuilder.build())
  }

  companion object {
    private const val TAG = "AGFcmMessagingService"
    private const val CHANNEL_ID = "gallery_high_priority_push_channel"
    private const val CONNECT_TIMEOUT_MS = 5000
    private const val READ_TIMEOUT_MS = 5000
  }
}
