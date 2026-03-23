import Foundation

extension WorkoutSessionViewModel {
    func buildSessionSummary(sessionVideoFileName: String? = nil) -> SessionSummary {
        let totalTime = Int(Date().timeIntervalSince(startTime))
        var calories = max(0, correctReps) * 4
        if initialMode == .plank {
            calories = Int(plankHold.elapsedSeconds * 0.5)
        }

        var items: [ExerciseSummaryItem] = []

        if initialMode == .plank {
            items.append(
                .isometric(
                    name: "Plank",
                    durationSeconds: plankHold.elapsedSeconds,
                    targetSeconds: plankHold.targetSeconds,
                    errors: buildErrorCounts(from: plankErrors.errorCounts),
                    mistakes: plankErrors.mistakeEvents
                )
            )
        } else if initialMode == .lunges {
            items.append(
                .movement(
                    name: "Lunges",
                    totalReps: totalReps,
                    correctReps: correctReps,
                    incorrectReps: incorrectReps,
                    targetCorrectReps: lungesTargetCorrectReps,
                    errors: buildErrorCounts(from: lungesErrors.errorCounts),
                    mistakes: lungesErrors.mistakeEvents
                )
            )
        } else {
            items.append(
                .isometric(
                    name: "Wall-Sit",
                    durationSeconds: wallSitHold.elapsedSeconds,
                    targetSeconds: wallSitHold.targetSeconds,
                    errors: buildErrorCounts(from: wallSitErrors.errorCounts),
                    mistakes: wallSitErrors.mistakeEvents
                )
            )

            items.append(
                .movement(
                    name: "Squat",
                    totalReps: totalReps,
                    correctReps: correctReps,
                    incorrectReps: incorrectReps,
                    targetCorrectReps: squatTargetCorrectReps,
                    errors: buildErrorCounts(from: squatErrors.errorCounts),
                    mistakes: squatErrors.mistakeEvents
                )
            )

            items.append(
                .isometric(
                    name: "Plank",
                    durationSeconds: plankHold.elapsedSeconds,
                    targetSeconds: plankHold.targetSeconds,
                    errors: buildErrorCounts(from: plankErrors.errorCounts),
                    mistakes: plankErrors.mistakeEvents
                )
            )
        }

        return SessionSummary(
            items: items,
            totalTimeSeconds: totalTime,
            estimatedCalories: calories,
            sessionVideoFileName: sessionVideoFileName
        )
    }

    func buildErrorCounts(from counts: [String: Int]) -> [ErrorCount] {
        counts
            .filter { $0.value > 0 }
            .map {
                ErrorCount(
                    reason: Self.makeDisplayLabel(for: $0.key),
                    count: $0.value
                )
            }
            .sorted { $0.count > $1.count }
    }
}
