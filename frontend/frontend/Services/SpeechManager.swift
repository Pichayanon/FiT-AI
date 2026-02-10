import Foundation
import AVFoundation

/// Text-to-speech helper used across workout sessions.
final class SpeechManager: ObservableObject {
    private let speaker = AVSpeechSynthesizer()
    private var lastSpoken: String = ""
    private var lastSpokenAt: Date = .distantPast

    func configureAudioSession() {
        do {
            let session = AVAudioSession.sharedInstance()
            // playback + spokenAudio ช่วยให้เสียงออกแม้ silent หลายกรณี
            try session.setCategory(.playback, mode: .spokenAudio, options: [.duckOthers])
            try session.setActive(true)
        } catch {
            print("Audio session error: \(error)")
        }
    }

    /// ล้างข้อความ: เอาตัวเลข/สัญลักษณ์ confidence ออก
    func cleanForSpeech(_ text: String) -> String {
        var s = text.trimmingCharacters(in: .whitespacesAndNewlines)

        // ตัดหลัง "•" (เช่น "knees_in • 0.92")
        if let dot = s.firstIndex(of: "•") {
            s = String(s[..<dot])
        }

        // ตัดหลัง "(" (เช่น "knees_in (0.923)")
        if let paren = s.firstIndex(of: "(") {
            s = String(s[..<paren])
        }

        // เอาตัวเลข/จุด/เปอร์เซ็นต์ ออก
        s = s.replacingOccurrences(
            of: #"[\d\.\%\-\+]+"#,
            with: "",
            options: .regularExpression
        )

        // ปรับ underscore -> เว้นวรรค
        s = s.replacingOccurrences(of: "_", with: " ")

        // เก็บเฉพาะตัวอักษร + เว้นวรรค (รองรับไทย/อังกฤษ)
        s = s.replacingOccurrences(
            of: #"[^A-Za-zก-๙\s]+"#,
            with: "",
            options: .regularExpression
        )

        // จัดช่องว่างซ้ำ
        s = s.replacingOccurrences(
            of: #"\s+"#,
            with: " ",
            options: .regularExpression
        )

        return s.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// เงื่อนไขว่าจะพูดไหม (กันรัว)
    func shouldSpeak(_ rawText: String) -> Bool {
        let t = rawText.lowercased()
        if t.isEmpty { return false }
        if t.contains("correct") { return false } // ไม่พูดถ้า correct
        if t.contains("hold") { return false }    // กันพูด "hold..." รัว
        // "passed" ให้พูดได้
        return true
    }

    func speak(
        _ text: String,
        language: String = "en-US",
        minInterval: TimeInterval = 1.2,
        allowRepeat: Bool = false
    ) {
        let clean = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !clean.isEmpty else { return }

        // กันพูดซ้ำ (ถ้าไม่ได้ allowRepeat)
        if !allowRepeat, clean == lastSpoken { return }

        // กันพูดถี่เกิน (ถ้า minInterval > 0)
        let now = Date()
        if minInterval > 0, now.timeIntervalSince(lastSpokenAt) < minInterval { return }

        lastSpoken = clean
        lastSpokenAt = now

        let u = AVSpeechUtterance(string: clean)
        u.voice = AVSpeechSynthesisVoice(language: language)
        u.rate = 0.50
        u.pitchMultiplier = 1.0
        u.volume = 1.0

        speaker.stopSpeaking(at: .immediate)
        speaker.speak(u)
    }

    func stop() {
        speaker.stopSpeaking(at: .immediate)
    }
}

