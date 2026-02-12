import Foundation
import SwiftUI

@MainActor
final class ProgressViewModel: ObservableObject {
    @Published var isLoading = false
    @Published var errorMessage: String?
    @Published var historyItems: [WorkoutHistoryItem] = []
    @Published var calorieStats: [DailyCalorie] = []
    @Published var workoutSummaries: [WorkoutSetSummary] = []

    private let workoutHistory = WorkoutHistoryService()
    private let calendar = Calendar.current

    init() {}

    func load() async {
        isLoading = true
        errorMessage = nil
        defer { isLoading = false }
        do {
            let sessions = try await workoutHistory.fetchSessions(limit: 50)
            historyItems = sessions.map { id, record in
                WorkoutHistoryItem(
                    id: id,
                    setTitle: record.setTitle,
                    completedAt: Date(timeIntervalSince1970: record.completedAtSeconds),
                    totalTimeSeconds: record.totalTimeSeconds,
                    estimatedCalories: record.estimatedCalories,
                    record: record
                )
            }
            calorieStats = computeCalorieStats()
            workoutSummaries = computeWorkoutSummaries()
        } catch {
            errorMessage = error.localizedDescription
            historyItems = []
            calorieStats = emptyCalorieStats()
            workoutSummaries = []
        }
    }

    private func emptyCalorieStats() -> [DailyCalorie] {
        let today = calendar.startOfDay(for: Date())
        return (0..<7).compactMap { offset in
            calendar.date(byAdding: .day, value: -offset, to: today).map { DailyCalorie(date: $0, calories: 0) }
        }.reversed()
    }

    private func computeCalorieStats() -> [DailyCalorie] {
        let today = calendar.startOfDay(for: Date())
        var dayCalories: [Date: Int] = [:]
        for offset in 0..<7 {
            guard let day = calendar.date(byAdding: .day, value: -offset, to: today) else { continue }
            dayCalories[day] = 0
        }
        for item in historyItems {
            let day = calendar.startOfDay(for: item.completedAt)
            dayCalories[day, default: 0] += item.estimatedCalories
        }
        return (0..<7).compactMap { offset in
            calendar.date(byAdding: .day, value: -offset, to: today).map { DailyCalorie(date: $0, calories: dayCalories[$0] ?? 0) }
        }.reversed()
    }

    private func computeWorkoutSummaries() -> [WorkoutSetSummary] {
        let grouped = Dictionary(grouping: historyItems, by: { $0.setTitle })
        return grouped.map { name, items in
            WorkoutSetSummary(
                name: name,
                timesCompleted: items.count,
                totalCalories: items.map(\.estimatedCalories).reduce(0, +)
            )
        }.sorted { $0.timesCompleted > $1.timesCompleted }
    }
}
