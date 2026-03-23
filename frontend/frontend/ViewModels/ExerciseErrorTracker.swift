import Foundation

struct ExerciseErrorTracker {
    var errorCounts: [String: Int] = [:]
    var mistakeEvents: [MistakeEvent] = []
    var lastPredictionWasCorrect: Bool = true
    private var recordedMovementMistakeKeys: Set<String> = []

    mutating func recordTransitionMistake(
        prediction: String,
        elapsedSeconds: Double,
        repNumber: Int?,
        displayLabel: String
    ) {
        if lastPredictionWasCorrect {
            errorCounts[prediction] = (errorCounts[prediction] ?? 0) + 1
            mistakeEvents.append(
                MistakeEvent(
                    atSecond: elapsedSeconds,
                    reason: displayLabel,
                    repNumber: repNumber
                )
            )
        }
        lastPredictionWasCorrect = false
    }

    mutating func recordMovementMistake(
        prediction: String,
        elapsedSeconds: Double,
        repNumber: Int?,
        displayLabel: String,
        eventKey: String
    ) {
        guard recordedMovementMistakeKeys.insert(eventKey).inserted else { return }

        errorCounts[prediction] = (errorCounts[prediction] ?? 0) + 1
        mistakeEvents.append(
            MistakeEvent(
                atSecond: elapsedSeconds,
                reason: displayLabel,
                repNumber: repNumber
            )
        )
        lastPredictionWasCorrect = false
    }

    mutating func markCorrectPrediction() {
        lastPredictionWasCorrect = true
    }

    mutating func reset() {
        errorCounts = [:]
        mistakeEvents = []
        lastPredictionWasCorrect = true
        recordedMovementMistakeKeys = []
    }
}
