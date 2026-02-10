import Foundation
import SwiftUI

@MainActor
final class WorkoutResultViewModel: ObservableObject {
    let totalReps: Int
    let correctReps: Int
    let incorrectReps: Int
    let totalTime: Int
    let estimatedCalories: Int

    init(totalReps: Int, correctReps: Int, incorrectReps: Int, totalTime: Int, estimatedCalories: Int) {
        self.totalReps = totalReps
        self.correctReps = correctReps
        self.incorrectReps = incorrectReps
        self.totalTime = totalTime
        self.estimatedCalories = estimatedCalories
    }

    var formattedTime: String {
        "\(totalTime / 60) min \(totalTime % 60) sec"
    }
}

