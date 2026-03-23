import Foundation

struct IsometricHoldState {
    var consecutiveCorrectCount: Int = 0
    var isCountingActive: Bool = false
    var isCurrentlyCorrect: Bool = false
    var elapsedSeconds: Double = 0
    var holdTargetSeconds: Double = 5.0
    var targetSeconds: Double = 5.0
    var hasPassed: Bool = false

    var progress: Double {
        guard targetSeconds > 0 else { return 0 }
        return min(1.0, max(0.0, elapsedSeconds / targetSeconds))
    }

    mutating func updateHoldTime(interval: Double) -> Bool {
        guard isCountingActive, !hasPassed, isCurrentlyCorrect else { return false }
        elapsedSeconds = min(targetSeconds, elapsedSeconds + interval)
        if elapsedSeconds >= targetSeconds {
            elapsedSeconds = targetSeconds
            hasPassed = true
            return true
        }
        return false
    }

    enum Action {
        case none
        case firstCorrectInStreak
        case gatePassed(feedbackText: String)
        case holdingProgress(feedbackText: String)
        case lostForm(feedbackText: String)
    }

    mutating func handlePrediction(
        isCorrect: Bool,
        incorrectFeedback: String
    ) -> Action {
        isCurrentlyCorrect = isCorrect

        if hasPassed { return .none }

        if !isCountingActive {
            if isCorrect {
                consecutiveCorrectCount = min(3, consecutiveCorrectCount + 1)
                if consecutiveCorrectCount == 1 {
                    return .firstCorrectInStreak
                }
                if consecutiveCorrectCount >= 3 {
                    isCountingActive = true
                    elapsedSeconds = 0
                    targetSeconds = holdTargetSeconds
                    return .gatePassed(feedbackText: "Start hold...")
                }
                return .holdingProgress(
                    feedbackText: "Correct \(consecutiveCorrectCount)/3 - Keep holding"
                )
            }

            if consecutiveCorrectCount != 0 {
                consecutiveCorrectCount = 0
                return .lostForm(feedbackText: incorrectFeedback)
            }
            return .none
        }

        if !isCorrect {
            isCountingActive = false
            isCurrentlyCorrect = false
            consecutiveCorrectCount = 0
            elapsedSeconds = 0
            return .lostForm(feedbackText: incorrectFeedback)
        }

        return .none
    }

    mutating func reset() {
        consecutiveCorrectCount = 0
        isCountingActive = false
        isCurrentlyCorrect = false
        elapsedSeconds = 0
        targetSeconds = holdTargetSeconds
        hasPassed = false
    }
}
