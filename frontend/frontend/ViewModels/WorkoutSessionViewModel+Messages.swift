import Foundation

extension WorkoutSessionViewModel {
    func handleBackendMessage(_ text: String) {
        guard let data = text.data(using: .utf8),
              let jsonObject = try? JSONSerialization.jsonObject(with: data),
              let payload = jsonObject as? [String: Any] else {
            return
        }

        let type = payload["type"] as? String ?? ""
        if type == "status" {
            handleStatusMessage(payload)
            return
        }

        if type == "info" {
            if let message = payload["message"] as? String {
                DispatchQueue.main.async {
                    let normalizedMessage = self.normalizeFeedbackText(message)
                    self.setFeedbackIfChanged(normalizedMessage)
                    self.speakBackendInfoIfNeeded(normalizedMessage)
                }
            }
            return
        }

        if type == "result" {
            handleResultMessage(payload)
        }
    }
}

private extension WorkoutSessionViewModel {
    func handleStatusMessage(_ payload: [String: Any]) {
        let stateRaw =
            (payload["state"] as? String ?? BackendPhase.noPose.rawValue).uppercased()
        let message = payload["message"] as? String
        let guidance = makeStatusGuidanceText(from: payload, stateRaw: stateRaw)

        DispatchQueue.main.async {
            self.backendState = stateRaw
            self.updateLiveSessionState(from: payload, stateRaw: stateRaw)
            self.applyStatusFeedback(message: message, guidance: guidance)
        }
    }

    func updateLiveSessionState(from payload: [String: Any], stateRaw: String) {
        if stateRaw == BackendPhase.noPose.rawValue {
            resetLiveStateForNoPose()
            return
        }

        switch mode {
        case .wallSit:
            updateWallSitStatus(from: payload, stateRaw: stateRaw)
        case .plank:
            updatePlankStatus(from: payload, stateRaw: stateRaw)
        case .lunges:
            updateLungesStatus(from: payload, stateRaw: stateRaw)
        case .squat:
            updateSquatStatus(stateRaw: stateRaw)
        }
    }

    func resetLiveStateForNoPose() {
        switch mode {
        case .wallSit:
            wallSitHold.isCurrentlyCorrect = false
            wallSitHold.isCountingActive = false
            wallSitHold.consecutiveCorrectCount = 0
            wallSitHold.elapsedSeconds = 0
            resetWallSitStartReminder()
        case .plank:
            plankHold.isCurrentlyCorrect = false
            plankHold.isCountingActive = false
            plankHold.consecutiveCorrectCount = 0
            plankHold.elapsedSeconds = 0
            resetPlankStartReminder()
        case .lunges:
            resetLungesStartReminder()
        case .squat:
            break
        }
    }

    func updateWallSitStatus(from payload: [String: Any], stateRaw: String) {
        if stateRaw == BackendPhase.havePose.rawValue,
           payload["standing"] as? Bool == true {
            if wallSitNeedsStartSince == nil {
                wallSitNeedsStartSince = Date()
                lastWallSitStartReminderAt = .distantPast
            }
            return
        }

        resetWallSitStartReminder()
    }

    func updatePlankStatus(from payload: [String: Any], stateRaw: String) {
        if stateRaw == BackendPhase.havePose.rawValue,
           payload["plank_pose_ok"] as? Bool == false {
            plankHold.isCurrentlyCorrect = false
            plankHold.isCountingActive = false
            plankHold.consecutiveCorrectCount = 0
            plankHold.elapsedSeconds = 0
            if plankNeedsStartSince == nil {
                plankNeedsStartSince = Date()
                lastPlankStartReminderAt = .distantPast
            }
            return
        }

        resetPlankStartReminder()
    }

    func updateLungesStatus(from payload: [String: Any], stateRaw: String) {
        if stateRaw == "READY", totalReps == 0 {
            if lungesNeedsStartSince == nil {
                lungesNeedsStartSince = Date()
                lastLungesStartReminderAt = .distantPast
            }
            return
        }

        resetLungesStartReminder()
    }

    func updateSquatStatus(stateRaw: String) {
        if stateRaw == "WAITING" {
            resetSquatStandIssueTracking()
        }
    }

    func applyStatusFeedback(message: String?, guidance: String?) {
        if let message {
            setFeedbackIfChanged(message)
            return
        }

        if let guidance {
            setFeedbackIfChanged(guidance)
        }
    }

    func handleResultMessage(_ payload: [String: Any]) {
        let prediction = (payload["prediction"] as? String ?? "...")
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        let confidence = payload["confidence"] as? Double
        let backendMode = payload["mode"] as? String ?? ""

        DispatchQueue.main.async {
            if !(self.mode == .squat && backendMode == "stand") {
                if let confidence {
                    self.setFeedbackIfChanged(
                        "\(prediction) - \(String(format: "%.2f", confidence))"
                    )
                } else {
                    self.setFeedbackIfChanged(prediction)
                }
            }

            switch self.mode {
            case .wallSit:
                self.handleWallSitPrediction(prediction, confidence: confidence)
            case .squat:
                self.handleSquatPrediction(prediction, confidence: confidence, payload: payload)
            case .plank:
                self.handlePlankPrediction(prediction, confidence: confidence)
            case .lunges:
                self.handleLungesPrediction(prediction, payload: payload)
            }
        }
    }
}
