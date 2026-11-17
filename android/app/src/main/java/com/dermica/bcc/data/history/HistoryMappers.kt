package com.dermica.bcc.data.history

/**
 * Valores crudos de verdict persistidos en `history_events.verdict`.
 *
 * Origen: contrato de la tabla (S6) — verdict se guarda como texto legible por SQL,
 * no como enum serializado; HistoryEvent los escribe en addImagePrediction y el
 * mapper los devuelve tal cual en HistoryItem.Image.verdict.
 */
internal const val VERDICT_BCC = "BCC"
internal const val VERDICT_NO_BCC = "NO_BCC"

/**
 * Mapea entidad Room → dominio del historial.
 *
 * Lugar en el flujo: datos (data/history), frontera tabla↔dominio. Lo consume
 * HistoryRepository (items) vía mapNotNull; la fila inválida queda fuera de la
 * lista. Cubierto por HistoryMappersTest en JVM.
 *
 * Devuelve null (y la fila se omite del historial) cuando la fila está
 * incompleta o corrupta: un evento persistido a medias por un crash no debe
 * romper la lista ni inventar datos que no existen (p.ej. un verdict NULL).
 * La política de recuperación (borrar la fila) se deja a una futura migración;
 * en T3 simplemente no se renderiza.
 *
 * @receiver HistoryEvent: fila cruda de history_events (Room).
 * @return HistoryItem?: dominio listo para la UI; null si la fila está
 *         incompleta/corrupta (IMAGE sin verdict/confidence, SUBSET con campos
 *         fuera de rango o NULL, o kind desconocido).
 *
 * Efectos: ninguno; es un mapeo puro.
 */
internal fun HistoryEvent.toItemOrNull(): HistoryItem? = when (kind) {
    HistoryKind.IMAGE -> {
        val storedVerdict = verdict?.trim()?.takeIf { it.isNotEmpty() } ?: return null
        val storedConfidence = confidence ?: return null
        HistoryItem.Image(
            id = id,
            createdAt = createdAt,
            imageName = imageName,
            verdict = storedVerdict,
            confidence = storedConfidence,
        )
    }

    HistoryKind.SUBSET -> {
        // Todos estos son obligatorios para renderizar la card de subconjunto.
        val folder = folderLabel?.takeIf { it.isNotBlank() } ?: return null
        val count = imageCount ?: return null
        val taValue = ta ?: return null
        val nfsValue = nfs ?: return null
        val peoValue = peo ?: return null
        val totalTp = tp ?: return null
        val totalTn = tn ?: return null
        val totalFp = fp ?: return null
        val totalFn = fn ?: return null
        HistoryItem.Subset(
            id = id,
            createdAt = createdAt,
            folderLabel = folder,
            imageCount = count,
            excludedCount = excludedCount ?: 0,
            ta = taValue,
            nfs = nfsValue.toInt(),
            peo = peoValue,
            tp = totalTp,
            tn = totalTn,
            fp = totalFp,
            fn = totalFn,
        )
    }

    else -> null
}
