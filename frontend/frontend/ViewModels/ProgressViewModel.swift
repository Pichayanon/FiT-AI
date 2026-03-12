import Foundation
import SwiftUI

/// Loads workout history and derives the progress-screen aggregates from it.
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
            historyItems = buildHistoryItems(from: sessions)
            calorieStats = computeCalorieStats()
            workoutSummaries = computeWorkoutSummaries()
        } catch {
            errorMessage = error.localizedDescription
            historyItems = []
            calorieStats = emptyCalorieStats()
            workoutSummaries = []
        }
    }

    private func buildHistoryItems(
        from sessions: [(id: String, record: WorkoutSessionRecord)]
    ) -> [WorkoutHistoryItem] {
        sessions.map { id, record in
            WorkoutHistoryItem(
                id: id,
                setTitle: record.setTitle,
                completedAt: Date(timeIntervalSince1970: record.completedAtSeconds),
                totalTimeSeconds: record.totalTimeSeconds,
                estimatedCalories: record.estimatedCalories,
                record: record
            )
        }
    }

    private func recentDays(count: Int) -> [Date] {
        let today = calendar.startOfDay(for: Date())
        return (0..<count).compactMap { offset in
            calendar.date(byAdding: .day, value: -offset, to: today)
        }.reversed()
    }

    private func emptyCalorieStats() -> [DailyCalorie] {
        recentDays(count: 7).map { DailyCalorie(date: $0, calories: 0) }
    }

    private func computeCalorieStats() -> [DailyCalorie] {
        let days = recentDays(count: 7)
        var dayCalories: [Date: Int] = [:]
        for day in days {
            dayCalories[day] = 0
        }
        for item in historyItems {
            let day = calendar.startOfDay(for: item.completedAt)
            dayCalories[day, default: 0] += item.estimatedCalories
        }
        return days.map { DailyCalorie(date: $0, calories: dayCalories[$0] ?? 0) }
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
