import Foundation
import SwiftUI

@MainActor
final class DailyDetailViewModel: ObservableObject {
    let date: Date
    let workoutSets: [WorkoutSetSummary]

    var totalCalories: Int {
        workoutSets.map { $0.totalCalories }.reduce(0, +)
    }

    init(date: Date) {
        self.date = date
        self.workoutSets = [
            .init(name: "Beginner Level 1", timesCompleted: 1, totalCalories: 150),
            .init(name: "Beginner Level 2", timesCompleted: 1, totalCalories: 120)
        ]
    }
}

