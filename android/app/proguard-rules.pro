# Reglas R8 (release). LiteRT carga clases por nombre vía JNI; se conservan para no romper la inferencia.
-keep class com.google.ai.edge.litert.** { *; }
-keep class org.tensorflow.lite.** { *; }
