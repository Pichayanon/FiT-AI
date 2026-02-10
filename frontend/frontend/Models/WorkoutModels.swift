import Foundation

// MARK: - Progress / Summary models

struct DailyCalorie: Identifiable {
    let id = UUID()
    let date: Date
    let calories: Int
}

struct WorkoutSetSummary: Identifiable {
    let id = UUID()
    let name: String
    let timesCompleted: Int
    let totalCalories: Int
}

// MARK: - Workout detail models

struct WorkoutExercise: Identifiable {
    let id = UUID()
    let name: String
    let imageName: String
    let reps: String
}

