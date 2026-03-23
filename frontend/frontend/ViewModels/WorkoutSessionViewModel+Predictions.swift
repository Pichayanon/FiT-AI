import Foundation

private struct MovementPredictionRequest {
    let prediction: String
    let payload: [String: Any]
    let goodPrefix: String
    let targetReps: Int
    let onTargetReached: () -> Void
}

private struct MovementRepCounts {
    let total: Int?
    let correct: Int?
    let incorrect: Int?
}

private func readInt(_ value: Any?) -> Int? {
    (value as? Int) ?? (value as? Double).map(Int.init)
}

extension WorkoutSessionViewModel {
    func handleWallSitPrediction(
        _ prediction: String,
        confidence _: Double?
    ) {
        resetWallSitStartReminder()

        let isCorrectPrediction = prediction.lowercased().contains("correct")
        if isCorrectPrediction {
            wallSitErrors.markCorrectPrediction()
        } else if shouldTrackMistakeLabel(prediction) {
            wallSitErrors.recordTransitionMistake(
                prediction: prediction,
                elapsedSeconds: sessionElapsedSeconds(),
                repNumber: nil,
                displayLabel: Self.makeDisplayLabel(for: prediction)
            )
        }

        if passedWallSit { return }

        let action = wallSitHold.handlePrediction(
            isCorrect: isCorrectPrediction,
            incorrectFeedback: makeIncorrectFeedbackText(for: prediction)
        )
        applyHoldAction(action)
    }

    func handlePlankPrediction(
        _ prediction: String,
        confidence _: Double?
    ) {
        let isCorrectPrediction = prediction.lowercased().contains("correct")
        if isCorrectPrediction {
            plankErrors.markCorrectPrediction()
        } else if shouldTrackMistakeLabel(prediction) {
            plankErrors.recordTransitionMistake(
                prediction: prediction,
                elapsedSeconds: sessionElapsedSeconds(),
                repNumber: nil,
                displayLabel: Self.makeDisplayLabel(for: prediction)
            )
        }

        if passedPlank { return }

        let action = plankHold.handlePrediction(
            isCorrect: isCorrectPrediction,
            incorrectFeedback: makeIncorrectFeedbackText(for: prediction)
        )
        applyHoldAction(action)
    }

    func handleSquatPrediction(
        _ prediction: String,
        confidence _: Double?,
        payload: [String: Any]
    ) {
        let backendMode = payload["mode"] as? String ?? ""
        if backendMode == "stand" {
            handleSquatStandPrediction(prediction, payload: payload)
            return
        }

        resetSquatStandIssueTracking()
        squatStarted = true

        let request = MovementPredictionRequest(
            prediction: prediction,
            payload: payload,
            goodPrefix: "good",
            targetReps: squatTargetCorrectReps,
            onTargetReached: { [weak self] in
                guard let self else { return }
                self.speech.speak(
                    "Passed",
                    language: self.speechLang,
                    minInterval: 0,
                    deliveryMode: .enqueue
                )
                self.cameraManager.stopStreaming()
                self.startTransitionCountdown(for: .plank)
            }
        )
        handleMovementPrediction(request, tracker: &squatErrors)
    }

    func handleLungesPrediction(
        _ prediction: String,
        payload: [String: Any]
    ) {
        resetLungesStartReminder()

        let request = MovementPredictionRequest(
            prediction: prediction,
            payload: payload,
            goodPrefix: "correct",
            targetReps: lungesTargetCorrectReps,
            onTargetReached: { [weak self] in
                guard let self else { return }
                self.setFeedbackIfChanged("Passed")
                self.finishSession(completionSpeech: "Passed")
            }
        )
        handleMovementPrediction(request, tracker: &lungesErrors)
    }

    func applyHoldAction(_ action: IsometricHoldState.Action) {
        switch action {
        case .none:
            break
        case .firstCorrectInStreak:
            speech.speak("OK", language: speechLang, minInterval: 0)
        case .gatePassed(let feedbackText),
             .holdingProgress(let feedbackText),
             .lostForm(let feedbackText):
            setFeedbackIfChanged(feedbackText)
        }
    }
}

private extension WorkoutSessionViewModel {
    func handleSquatStandPrediction(
        _ prediction: String,
        payload: [String: Any]
    ) {
        let standOk = payload["stand_ok"] as? Bool ?? false
        if standOk {
            squatErrors.markCorrectPrediction()
            setFeedbackIfChanged("Stand OK")
            resetSquatStandIssueTracking()
            if !squatStandOK {
                squatStandOK = true
                lastStandOKAt = Date()
                lastPleaseStartAt = .distantPast
                speech.speak("Stand OK", language: speechLang, minInterval: 0)
            }
            return
        }

        let standFeedback = Self.makeDisplayLabel(for: prediction)
        setFeedbackIfChanged(standFeedback)
        squatStandOK = false
        lastStandOKAt = nil
        lastPleaseStartAt = .distantPast

        if shouldRepeatSquatStandFeedback(standFeedback) {
            startSquatStandIssueReminder(for: standFeedback)
            return
        }

        resetSquatStandIssueTracking()
        speech.speak(standFeedback, language: speechLang, minInterval: 2.0)
    }

    func handleMovementPrediction(
        _ request: MovementPredictionRequest,
        tracker: inout ExerciseErrorTracker
    ) {
        let counts = extractMovementRepCounts(from: request.payload)
        syncMovementTotals(with: counts)

        let currentRepNumber = makeCurrentRepNumber(from: counts)
        let eventKey = makeMovementEventKey(
            from: request.payload,
            prediction: request.prediction,
            currentRepNumber: currentRepNumber
        )

        recordMovementPrediction(
            request.prediction,
            tracker: &tracker,
            goodPrefix: request.goodPrefix,
            currentRepNumber: currentRepNumber,
            eventKey: eventKey
        )
        syncMovementRepCounts(
            with: counts,
            targetReps: request.targetReps,
            onTargetReached: request.onTargetReached
        )
    }

    func extractMovementRepCounts(from payload: [String: Any]) -> MovementRepCounts {
        let reps = payload["reps"] as? [String: Any]
        return MovementRepCounts(
            total: readInt(reps?["total"]),
            correct: readInt(reps?["correct"]) ?? readInt(reps?["good"]),
            incorrect: readInt(reps?["incorrect"]) ?? readInt(reps?["bad"])
        )
    }

    func syncMovementTotals(with counts: MovementRepCounts) {
        if let total = counts.total ?? combinedRepTotal(from: counts) {
            totalReps = total
        }
    }

    func combinedRepTotal(from counts: MovementRepCounts) -> Int? {
        guard let correct = counts.correct, let incorrect = counts.incorrect else {
            return nil
        }
        return correct + incorrect
    }

    func makeCurrentRepNumber(from counts: MovementRepCounts) -> Int? {
        let repNumber = counts.total ?? combinedRepTotal(from: counts) ?? totalReps
        return repNumber > 0 ? repNumber : nil
    }

    func makeMovementEventKey(
        from payload: [String: Any],
        prediction: String,
        currentRepNumber: Int?
    ) -> String {
        if let eventIndex = readInt(payload["event_i"]) {
            return "event:\(eventIndex)"
        }
        if let currentRepNumber {
            return "rep:\(currentRepNumber)"
        }
        let decisecond = Int((sessionElapsedSeconds() * 10).rounded())
        return "time:\(decisecond):\(prediction)"
    }

    func recordMovementPrediction(
        _ prediction: String,
        tracker: inout ExerciseErrorTracker,
        goodPrefix: String,
        currentRepNumber: Int?,
        eventKey: String
    ) {
        if prediction.hasPrefix(goodPrefix) {
            tracker.markCorrectPrediction()
            return
        }

        if shouldTrackMistakeLabel(prediction) {
            tracker.recordMovementMistake(
                prediction: prediction,
                elapsedSeconds: sessionElapsedSeconds(),
                repNumber: currentRepNumber,
                displayLabel: Self.makeDisplayLabel(for: prediction),
                eventKey: eventKey
            )
            speakMovementMistakeIfNeeded(prediction)
        }
    }

    func syncMovementRepCounts(
        with counts: MovementRepCounts,
        targetReps: Int,
        onTargetReached: () -> Void
    ) {
        if let correct = counts.correct {
            if correct > correctReps {
                speech.speak(
                    "Good",
                    language: speechLang,
                    minInterval: 0.5,
                    allowRepeat: true
                )
                lastSpokenCorrectReps = correct
            }
            correctReps = correct
        }

        if let incorrect = counts.incorrect {
            incorrectReps = incorrect
        }

        if correctReps >= targetReps {
            onTargetReached()
        }
    }
}
