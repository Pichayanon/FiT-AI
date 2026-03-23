import Foundation

extension WorkoutSessionViewModel {
    static func makeDisplayLabel(for backendLabel: String) -> String {
        let normalizedLabel = backendLabel.lowercased()
        if normalizedLabel.contains("feet_too_close")
            || normalizedLabel.contains("feet too close") {
            return "Feet too close"
        }
        if normalizedLabel.contains("knee_ins")
            || normalizedLabel.contains("knees_in")
            || normalizedLabel.contains("knees in") {
            return "Knees in"
        }
        if normalizedLabel.contains("round_back") {
            return "Round back"
        }
        if normalizedLabel.contains("torso_lean_forward")
            || normalizedLabel.contains("torso lean forward") {
            return "Torso lean forward"
        }
        if normalizedLabel.contains("knee_over_toe")
            || normalizedLabel.contains("knee over toe") {
            return "Knee over toe"
        }
        if normalizedLabel.contains("not_deep_enough") {
            return "Not deep enough"
        }
        if normalizedLabel.contains("stand_too_narrow") {
            return "Stand too narrow"
        }
        if normalizedLabel.contains("stand_too_wide") {
            return "Stand too wide"
        }

        let labelWithSpaces = backendLabel.replacingOccurrences(of: "_", with: " ")
        guard !labelWithSpaces.isEmpty else { return backendLabel }
        return labelWithSpaces
            .split(separator: " ")
            .map { $0.prefix(1).uppercased() + $0.dropFirst().lowercased() }
            .joined(separator: " ")
    }

    func sessionElapsedSeconds() -> Double {
        max(0, Date().timeIntervalSince(startTime))
    }

    func shouldTrackMistakeLabel(_ prediction: String) -> Bool {
        let normalizedPrediction =
            prediction.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if normalizedPrediction.isEmpty || normalizedPrediction == "..." {
            return false
        }
        if normalizedPrediction.contains("correct") {
            return false
        }
        if normalizedPrediction == "good"
            || normalizedPrediction == "passed"
            || normalizedPrediction == "stand"
            || normalizedPrediction == "good_stand"
            || normalizedPrediction == "good_squat" {
            return false
        }
        return true
    }

    func makeIncorrectFeedbackText(for prediction: String) -> String {
        let trimmedPrediction = prediction.trimmingCharacters(
            in: .whitespacesAndNewlines
        )
        guard !trimmedPrediction.isEmpty, trimmedPrediction != "..." else {
            return "Adjust your position"
        }
        return Self.makeDisplayLabel(for: trimmedPrediction)
    }

    func makeStatusGuidanceText(
        from payload: [String: Any],
        stateRaw: String
    ) -> String? {
        if stateRaw == BackendPhase.noPose.rawValue {
            guard let tooDark = payload["too_dark"] as? Bool else { return nil }
            return tooDark
                ? "Please adjust your lights."
                : makeFullBodyGuidanceText()
        }

        if mode == .plank,
           stateRaw == BackendPhase.havePose.rawValue,
           let plankPoseOK = payload["plank_pose_ok"] as? Bool,
           !plankPoseOK {
            return "Lower into plank position"
        }

        if (mode == .lunges || mode == .squat) && stateRaw == "WAITING" {
            if payload["too_dark"] as? Bool == true {
                return "Please adjust your lights."
            }
            return makeFullBodyGuidanceText()
        }

        return nil
    }

    func shouldRepeatSquatStandFeedback(_ feedback: String) -> Bool {
        let normalizedFeedback = feedback.lowercased()
        return normalizedFeedback.contains("stand too narrow")
            || normalizedFeedback.contains("stand too wide")
    }

    func startSquatStandIssueReminder(for feedback: String) {
        guard shouldRepeatSquatStandFeedback(feedback) else {
            resetSquatStandIssueTracking()
            return
        }

        let now = Date()
        guard activeRepeatableSquatStandIssue != feedback else { return }

        activeRepeatableSquatStandIssue = feedback
        activeRepeatableSquatStandIssueSince = now
        lastRepeatableSquatStandIssueSpokenAt = now
        speech.speak(
            feedback,
            language: speechLang,
            minInterval: 0,
            allowRepeat: true
        )
    }

    func resetSquatStandIssueTracking() {
        activeRepeatableSquatStandIssue = nil
        activeRepeatableSquatStandIssueSince = nil
        lastRepeatableSquatStandIssueSpokenAt = .distantPast
    }

    func resetPlankStartReminder() {
        plankNeedsStartSince = nil
        lastPlankStartReminderAt = .distantPast
    }

    func resetWallSitStartReminder() {
        wallSitNeedsStartSince = nil
        lastWallSitStartReminderAt = .distantPast
    }

    func resetLungesStartReminder() {
        lungesNeedsStartSince = nil
        lastLungesStartReminderAt = .distantPast
    }

    func makeFullBodyGuidanceText() -> String {
        switch mode {
        case .squat:
            return "Adjust your camera to show your full body from the front"
        case .wallSit, .plank, .lunges:
            return "Adjust your camera to show your full body from the side"
        }
    }

    func normalizeFeedbackText(_ raw: String) -> String {
        let normalized = raw
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        if normalized.contains("adjust your camera") {
            return makeFullBodyGuidanceText()
        }
        if normalized.contains("full body") {
            return makeFullBodyGuidanceText()
        }
        return raw
    }

    func speakBackendInfoIfNeeded(_ text: String) {
        let cleaned = speech.cleanForSpeech(text)
        guard speech.shouldSpeak(cleaned) else { return }
        speech.speak(
            cleaned,
            language: speechLang,
            minInterval: 0,
            deliveryMode: makeSpeechDeliveryMode(for: cleaned)
        )
    }

    func makeSpeechDeliveryMode(
        for feedbackText: String
    ) -> SpeechService.DeliveryMode {
        let normalized = feedbackText.lowercased()
        if normalized.contains("adjust your camera") {
            return .enqueue
        }
        if normalized.contains("full body") {
            return .enqueue
        }
        if normalized.contains("adjust your lights") {
            return .enqueue
        }
        if normalized.contains("plank position") {
            return .enqueue
        }
        if normalized.contains("side view ok") {
            return .interrupt
        }
        if normalized.contains("front view ok") {
            return .interrupt
        }
        return .dropIfBusy
    }

    func speakMovementMistakeIfNeeded(_ prediction: String) {
        let feedbackText = makeIncorrectFeedbackText(for: prediction)
        guard speech.shouldSpeak(feedbackText) else { return }
        speech.speak(
            feedbackText,
            language: speechLang,
            minInterval: 1.2,
            allowRepeat: true
        )
    }

    func setFeedbackIfChanged(_ newText: String) {
        if feedback != newText {
            feedback = newText
        }
    }
}
