import Foundation
import SwiftUI

@MainActor
final class ProgressViewModel: ObservableObject {
    @Published var calorieStats: [DailyCalorie]
    @Published var workoutSummaries: [WorkoutSetSummary]

    init() {
        self.calorieStats = [
            .init(date: Date().addingTimeInterval(-6 * 86400), calories: 120),
            .init(date: Date().addingTimeInterval(-5 * 86400), calories: 150),
            .init(date: Date().addingTimeInterval(-4 * 86400), calories: 200),
            .init(date: Date().addingTimeInterval(-3 * 86400), calories: 180),
            .init(date: Date().addingTimeInterval(-2 * 86400), calories: 160),
            .init(date: Date().addingTimeInterval(-1 * 86400), calories: 210),
            .init(date: Date(), calories: 190)
        ]

        self.workoutSummaries = [
            .init(name: "Beginner Level 1", timesCompleted: 3, totalCalories: 450),
            .init(name: "Beginner Level 2", timesCompleted: 2, totalCalories: 320),
            .init(name: "Intermediate Level 1", timesCompleted: 1, totalCalories: 200)
        ]
    }
}

