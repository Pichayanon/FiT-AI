import Foundation
import SwiftUI

extension WorkoutSessionViewModel {
    func onAppear() {
        speech.configureAudioSession()
        cameraManager.startSession()
        cameraManager.onBackendMessage = { [weak self] text in
            self?.handleBackendMessage(text)
        }
    }

    func onDisappear() {
        stopSession()
        cameraManager.stopSession(discardRecording: !navigateToResult)
    }

    func startSession() {
        guard !isFinalizingSession else { return }

        withAnimation(.easeInOut(duration: 0.2)) {
            isSessionRunning = true
        }

        startTime = Date()
        isFinalizingSession = false
        backendState = BackendPhase.noPose.rawValue

        resetSessionState()
        feedback = mode == .squat
            ? "Streaming to backend..."
            : "Listen and get in position..."

        if isSessionRecordingEnabled {
            cameraManager.beginSessionRecording()
        }
        speech.speak(
            "Session started",
            language: speechLang,
            minInterval: 0,
            deliveryMode: .enqueue
        )
        showExerciseIntroIfNeeded(for: mode) { [weak self] in
            self?.startStreamingForCurrentMode()
        }
    }

    func stopSession() {
        isSessionRunning = false
        cameraManager.stopStreaming()
        backendState = BackendPhase.noPose.rawValue
        showExerciseIntroOverlay = false
        showSquatPreview = false
        showPlankPreview = false
        resetPlankStartReminder()
        speech.stop()
    }

    func finishSession(completionSpeech: String = "Session complete") {
        guard isSessionRunning, !isFinalizingSession else { return }

        isFinalizingSession = true
        stopSession()
        speech.speak(completionSpeech, language: speechLang, minInterval: 0)
        feedback = "Saving summary..."

        if isSessionRecordingEnabled {
            cameraManager.finishSessionRecording { [weak self] url in
                guard let self else { return }

                let videoFileName = url?.lastPathComponent
                self.sessionSummary = self.buildSessionSummary(
                    sessionVideoFileName: videoFileName
                )
                Task {
                    await self.workoutHistory.saveSession(
                        summary: self.sessionSummary,
                        setTitle: self.setTitle
                    )
                }
                self.isFinalizingSession = false
                self.navigateToResult = true
            }
            return
        }

        sessionSummary = buildSessionSummary(sessionVideoFileName: nil)
        Task {
            await workoutHistory.saveSession(
                summary: sessionSummary,
                setTitle: setTitle
            )
        }
        isFinalizingSession = false
        navigateToResult = true
    }
}

private extension WorkoutSessionViewModel {
    func resetSessionState() {
        mode = initialMode

        didSwitchToSquat = false
        didSwitchToPlank = false
        passedWallSit = false

        showExerciseIntroOverlay = false
        showSquatPreview = false
        squatPreviewSeconds = previewDurationSeconds
        showPlankPreview = false
        plankPreviewSeconds = previewDurationSeconds

        totalReps = 0
        correctReps = 0
        incorrectReps = 0
        lastSpokenCorrectReps = 0

        wallSitHold.reset()
        plankHold.reset()
        passedPlank = false

        squatStandOK = false
        squatStarted = false
        lastStandOKAt = nil
        lastPleaseStartAt = .distantPast
        resetSquatStandIssueTracking()
        resetWallSitStartReminder()
        resetPlankStartReminder()
        resetLungesStartReminder()

        wallSitErrors.reset()
        squatErrors.reset()
        plankErrors.reset()
        lungesErrors.reset()
    }

    func startStreamingForCurrentMode() {
        guard isSessionRunning else { return }
        setFeedbackIfChanged("Streaming to backend...")
        cameraManager.startStreaming(to: activeWSURL)
    }

    func showExerciseIntroIfNeeded(
        for mode: WorkoutSessionMode,
        onComplete: @escaping () -> Void
    ) {
        guard mode != .squat else {
            onComplete()
            return
        }

        withAnimation(.easeInOut(duration: 0.2)) {
            showExerciseIntroOverlay = true
        }

        speech.speak(
            mode.demoInstructionText,
            language: speechLang,
            minInterval: 0,
            allowRepeat: true,
            deliveryMode: .enqueue
        ) { [weak self] in
            guard let self else { return }
            DispatchQueue.main.asyncAfter(
                deadline: .now() + self.introOverlayPostSpeechDelaySeconds
            ) { [weak self] in
                guard let self else { return }
                guard self.isSessionRunning, self.mode == mode else { return }

                withAnimation(.easeInOut(duration: 0.2)) {
                    self.showExerciseIntroOverlay = false
                }
                onComplete()
            }
        }
    }
}
