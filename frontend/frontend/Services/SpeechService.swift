import Foundation
import AVFoundation

/// Text-to-speech helper used during workout sessions to provide real-time voice feedback.
///
/// Features:
/// - Cleans raw backend text (strips confidence scores, symbols, numbers) before speaking.
/// - Throttles speech to prevent rapid-fire announcements.
/// - Deduplicates consecutive identical messages.
/// - Filters out non-actionable status messages (e.g., "correct", "hold", "streaming").
final class SpeechService: NSObject, ObservableObject, @unchecked Sendable {
    enum DeliveryMode {
        case dropIfBusy
        case enqueue
        case interrupt
    }

    private let speaker = AVSpeechSynthesizer()
    private var lastSpoken: String = ""
    private var lastSpokenAt: Date = .distantPast
    private var completionHandlers: [ObjectIdentifier: () -> Void] = [:]

    override init() {
        super.init()
        speaker.delegate = self
    }

    var isSpeaking: Bool {
        speaker.isSpeaking || speaker.isPaused
    }

    /// Configures the audio session for spoken audio playback (works even in silent mode).
    func configureAudioSession() {
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playback, mode: .spokenAudio, options: [.duckOthers])
            try session.setActive(true)
        } catch {
            print("[SpeechService] Audio session error: \(error)")
        }
    }

    /// Strips confidence scores, symbols, and numbers from backend text to produce clean speech.
    ///
    /// Examples:
    /// - "knees_in . 0.92" -> "knees in"
    /// - "round_back (0.85)" -> "round back"
    func cleanForSpeech(_ text: String) -> String {
        var s = text.trimmingCharacters(in: .whitespacesAndNewlines)

        // Strip text after bullet separator (e.g., "knees_in . 0.92")
        if let dot = s.firstIndex(of: "\u{2022}") {
            s = String(s[..<dot])
        }

        // Strip text after opening parenthesis (e.g., "knees_in (0.923)")
        if let paren = s.firstIndex(of: "(") {
            s = String(s[..<paren])
        }

        // Remove digits, dots, percent signs, and arithmetic operators
        s = s.replacingOccurrences(
            of: #"[\d\.%\-\+]+"#,
            with: "",
            options: .regularExpression
        )

        // Convert underscores to spaces for natural speech
        s = s.replacingOccurrences(of: "_", with: " ")

        // Keep only letters and spaces (supports Thai and English)
        s = s.replacingOccurrences(
            of: #"[^A-Za-z\u{0E01}-\u{0E59}\s]+"#,
            with: "",
            options: .regularExpression
        )

        // Collapse multiple spaces into one
        s = s.replacingOccurrences(
            of: #"\s+"#,
            with: " ",
            options: .regularExpression
        )

        return s.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Determines whether the given text should be spoken aloud.
    /// Filters out non-actionable status messages to reduce speech clutter.
    func shouldSpeak(_ rawText: String) -> Bool {
        let t = rawText.lowercased()
        if t.isEmpty { return false }
        if t.contains("websocket connected") { return false }
        if t.contains("streaming") { return false }
        if t.contains("recording") { return false }
        if t.contains("correct") { return false }
        if t.contains("hold") { return false }
        if t.contains("passed") { return false }
        return true
    }

    /// Speaks the given text with throttle and deduplication controls.
    ///
    /// - Parameters:
    ///   - text: The text to speak.
    ///   - language: BCP 47 language code (default: "en-US").
    ///   - minInterval: Minimum seconds between consecutive speech events.
    ///   - allowRepeat: If false, prevents speaking the same text twice in a row.
    ///   - deliveryMode: Whether to drop, queue, or interrupt while another utterance is active.
    ///   - completion: Called when this utterance finishes or is cancelled.
    func speak(
        _ text: String,
        language: String = "en-US",
        minInterval: TimeInterval = 1.2,
        allowRepeat: Bool = false,
        deliveryMode: DeliveryMode = .dropIfBusy,
        completion: (() -> Void)? = nil
    ) {
        let clean = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !clean.isEmpty else { return }

        // Prevent speaking identical text consecutively unless explicitly allowed
        if !allowRepeat, clean == lastSpoken { return }

        if deliveryMode == .dropIfBusy, isSpeaking {
            return
        }

        // Enforce minimum interval between speech events
        let now = Date()
        if minInterval > 0, now.timeIntervalSince(lastSpokenAt) < minInterval { return }

        if deliveryMode == .interrupt {
            completionHandlers.removeAll()
            speaker.stopSpeaking(at: .immediate)
        }

        lastSpoken = clean
        lastSpokenAt = now

        let u = AVSpeechUtterance(string: clean)
        u.voice = AVSpeechSynthesisVoice(language: language)
        u.rate = 0.50
        u.pitchMultiplier = 1.0
        u.volume = 1.0

        if let completion {
            completionHandlers[ObjectIdentifier(u)] = completion
        }

        speaker.speak(u)
    }

    /// Immediately stops any in-progress speech.
    func stop() {
        completionHandlers.removeAll()
        speaker.stopSpeaking(at: .immediate)
    }

    private func runCompletion(for utterance: AVSpeechUtterance) {
        let key = ObjectIdentifier(utterance)
        guard let completion = completionHandlers.removeValue(forKey: key) else { return }

        DispatchQueue.main.async {
            completion()
        }
    }
}

extension SpeechService: AVSpeechSynthesizerDelegate {
    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        runCompletion(for: utterance)
    }

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        runCompletion(for: utterance)
    }
}
