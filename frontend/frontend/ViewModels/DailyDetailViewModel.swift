import Foundation
import SwiftUI

/// Provides mock data for the daily detail screen (placeholder until connected to real data).
@MainActor
final class DailyDetailViewModel: ObservableObject {
    let date: Date
    let workoutSets: [WorkoutSetSummary]

    /// Total calories across all workout sets for this day.
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
