import Foundation

extension WorkoutSessionViewModel {
    func startTransitionCountdown(for nextMode: WorkoutSessionMode) {
        let isSquatTransition = nextMode == .squat

        if isSquatTransition {
            showSquatPreview = true
            squatPreviewSeconds = previewDurationSeconds
        } else {
            showPlankPreview = true
            plankPreviewSeconds = previewDurationSeconds
        }

        speech.speak(
            nextMode.demoInstructionText,
            language: speechLang,
            minInterval: 0,
            allowRepeat: true,
            deliveryMode: .enqueue
        )

        for second in 1...previewDurationSeconds {
            DispatchQueue.main.asyncAfter(deadline: .now() + Double(second)) { [weak self] in
                guard let self else { return }
                let remaining = max(0, self.previewDurationSeconds - second)
                if isSquatTransition {
                    if self.showSquatPreview {
                        self.squatPreviewSeconds = remaining
                    }
                    return
                }

                if self.showPlankPreview {
                    self.plankPreviewSeconds = remaining
                }
            }
        }

        DispatchQueue.main.asyncAfter(
            deadline: .now() + Double(previewDurationSeconds)
        ) { [weak self] in
            guard let self else { return }
            if isSquatTransition {
                self.showSquatPreview = false
                self.activateSquatMode()
                return
            }

            self.showPlankPreview = false
            self.activatePlankMode()
        }
    }

    func activateSquatMode() {
        guard isSessionRunning, !didSwitchToSquat else { return }

        didSwitchToSquat = true
        mode = .squat
        backendState = BackendPhase.noPose.rawValue
        setFeedbackIfChanged("PASSED (Switch to Squat)")

        totalReps = 0
        correctReps = 0
        incorrectReps = 0
        lastSpokenCorrectReps = 0

        squatStandOK = false
        squatStarted = false
        lastStandOKAt = nil
        lastPleaseStartAt = .distantPast
        resetSquatStandIssueTracking()

        wallSitHold.isCountingActive = false
        wallSitHold.isCurrentlyCorrect = false

        cameraManager.stopStreaming()
        cameraManager.startStreaming(to: activeWSURL)
    }

    func activatePlankMode() {
        guard isSessionRunning, !didSwitchToPlank else { return }

        didSwitchToPlank = true
        mode = .plank
        backendState = BackendPhase.noPose.rawValue
        setFeedbackIfChanged("PASSED (Switch to Plank)")

        plankHold.reset()
        passedPlank = false
        plankErrors.reset()
        resetPlankStartReminder()

        cameraManager.stopStreaming()
        cameraManager.startStreaming(to: activeWSURL)
    }
}
