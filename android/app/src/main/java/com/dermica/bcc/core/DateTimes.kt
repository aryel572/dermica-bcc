package com.dermica.bcc.core

import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter

/**
 * Fecha-hora corta para las cards del historial (S6).
 *
 * Lugar en el flujo: núcleo puro (java.time, sin Android). Lo consume
 * ui/history/HistoryScreen para rotular cada HistoryItem; lo cubre
 * DateTimesTest en JVM. Su salida visible es el texto de la card
 * («dd/MM/yyyy HH:mm»), nunca una fecha ISO cruda.
 *
 * Patrón fijo dd/MM/yyyy HH:mm en 24 h (convención peruana), independiente del
 * idioma del dispositivo: el historial es un registro clínico y el formato debe
 * ser estable entre sesiones y capturas de pantalla de la tesis.
 */
object DateTimes {

    // Patrón dd/MM/yyyy HH:mm: convención peruana en 24 h. Lo consume formatDateTime
    // en cada card del historial; DateTimesTest lo fija para evitar derivas.
    private val SHORT: DateTimeFormatter = DateTimeFormatter.ofPattern("dd/MM/yyyy HH:mm")

    /**
     * Formatea [epochMillis] en la zona horaria indicada.
     *
     * @param epochMillis — Long: milisegundos UTC desde epoch (así se persisten
     *        los eventos en history_events.createdAt).
     * @param zone — ZoneId: zona a la que se convierte el instante; por defecto,
     *        la del dispositivo (ZoneId.systemDefault()).
     * @return String dd/MM/yyyy HH:mm en 24 h, lista para las cards del historial.
     */
    fun formatDateTime(epochMillis: Long, zone: ZoneId = ZoneId.systemDefault()): String =
        SHORT.withZone(zone).format(Instant.ofEpochMilli(epochMillis))
}
