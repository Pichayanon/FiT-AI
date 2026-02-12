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

// MARK: - Workout result summary (หลังจบ session: wall-sit + squat)

/// ข้อผิดพลาดอย่างหนึ่งกับจำนวนครั้ง
struct ErrorCount: Identifiable {
    let id = UUID()
    let reason: String
    let count: Int
}

/// รายการสรุปของท่าออกกำลังกายหนึ่งท่า
enum ExerciseSummaryItem: Identifiable {
    case movement(name: String, totalReps: Int, correctReps: Int, incorrectReps: Int, targetCorrectReps: Int, errors: [ErrorCount])
    case isometric(name: String, durationSeconds: Double, targetSeconds: Double, errors: [ErrorCount])

    var id: String {
        switch self {
        case .movement(let name, _, _, _, _, _): return "movement-\(name)"
        case .isometric(let name, _, _, _): return "isometric-\(name)"
        }
    }

    var displayName: String {
        switch self {
        case .movement(let name, _, _, _, _, _): return name
        case .isometric(let name, _, _, _): return name
        }
    }
}

struct SessionSummary {
    let items: [ExerciseSummaryItem]
    let totalTimeSeconds: Int
    let estimatedCalories: Int
}

