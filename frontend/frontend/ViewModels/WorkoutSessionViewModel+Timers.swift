import Foundation

extension WorkoutSessionViewModel {
    func handleWallSitTick() {
        guard isSessionRunning, mode == .wallSit, !passedWallSit, !showSquatPreview else {
            return
        }

        let now = Date()
        if let reminderSince = wallSitNeedsStartSince,
           now.timeIntervalSince(reminderSince) >= 5.0,
           now.timeIntervalSince(lastWallSitStartReminderAt) >= 5.0 {
            speech.speak(
                "Please start wall sit",
                language: speechLang,
                minInterval: 0,
                allowRepeat: true,
                deliveryMode: .enqueue
            )
            lastWallSitStartReminderAt = now
        }

        if wallSitHold.updateHoldTime(interval: 0.1) {
            passedWallSit = true
            setFeedbackIfChanged("Passed")
            speech.speak(
                "Passed",
                language: speechLang,
                minInterval: 0,
                deliveryMode: .enqueue
            )
            cameraManager.stopStreaming()
            startTransitionCountdown(for: .squat)
        }
    }

    func handleSquatIdleTick() {
        guard isSessionRunning, mode == .squat else { return }
        let now = Date()

        let hasSetupIssueOverlay =
            feedback.localizedCaseInsensitiveContains("adjust your camera")
            || feedback.localizedCaseInsensitiveContains("adjust your lights")

        if squatStandOK, !squatStarted, !hasSetupIssueOverlay, let okAt = lastStandOKAt {
            if now.timeIntervalSince(okAt) >= 5.0,
               now.timeIntervalSince(lastPleaseStartAt) >= 5.0 {
                speech.speak(
                    "Please start squat",
                    language: speechLang,
                    minInterval: 0,
                    allowRepeat: true
                )
                lastPleaseStartAt = now
            }
        }

        guard !squatStandOK, !squatStarted else { return }
        guard let activeIssue = activeRepeatableSquatStandIssue,
              let issueSince = activeRepeatableSquatStandIssueSince else { return }

        if now.timeIntervalSince(issueSince) >= 5.0,
           now.timeIntervalSince(lastRepeatableSquatStandIssueSpokenAt) >= 5.0 {
            speech.speak(
                activeIssue,
                language: speechLang,
                minInterval: 0,
                allowRepeat: true
            )
            lastRepeatableSquatStandIssueSpokenAt = now
        }
    }

    func handleLungesSetupTick() {
        guard isSessionRunning, mode == .lunges else { return }
        let now = Date()

        if let reminderSince = lungesNeedsStartSince,
           now.timeIntervalSince(reminderSince) >= 5.0,
           now.timeIntervalSince(lastLungesStartReminderAt) >= 5.0 {
            speech.speak(
                "Please start lunges",
                language: speechLang,
                minInterval: 0,
                allowRepeat: true,
                deliveryMode: .enqueue
            )
            lastLungesStartReminderAt = now
        }
    }

    func handlePlankTick() {
        guard isSessionRunning, mode == .plank, !passedPlank else { return }
        let now = Date()

        if let reminderSince = plankNeedsStartSince,
           now.timeIntervalSince(reminderSince) >= 5.0,
           now.timeIntervalSince(lastPlankStartReminderAt) >= 5.0 {
            speech.speak(
                "Please start plank",
                language: speechLang,
                minInterval: 0,
                allowRepeat: true,
                deliveryMode: .enqueue
            )
            lastPlankStartReminderAt = now
        }

        if plankHold.updateHoldTime(interval: 0.1) {
            passedPlank = true
            setFeedbackIfChanged("Passed")
            finishSession(completionSpeech: "Passed")
        }
    }

    func handleFeedbackChange() {
        guard isSessionRunning else { return }
        let cleaned = speech.cleanForSpeech(feedback)
        guard speech.shouldSpeak(cleaned) else { return }
        speech.speak(
            cleaned,
            language: speechLang,
            minInterval: 1.2,
            deliveryMode: makeSpeechDeliveryMode(for: cleaned)
        )
    }
}
