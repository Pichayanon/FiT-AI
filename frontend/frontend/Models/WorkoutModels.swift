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

// MARK: - Firestore workout history (Codable for saving)

struct ErrorCountRecord: Codable {
    let reason: String
    let count: Int
}

struct ExerciseRecord: Codable {
    let name: String
    let type: String // "movement" | "isometric"
    let totalReps: Int?
    let correctReps: Int?
    let incorrectReps: Int?
    let targetCorrectReps: Int?
    let durationSeconds: Double?
    let targetSeconds: Double?
    let errors: [ErrorCountRecord]
}

struct WorkoutSessionRecord: Codable {
    let userId: String
    let setTitle: String
    let completedAtSeconds: Double
    let totalTimeSeconds: Int
    let estimatedCalories: Int
    let exercises: [ExerciseRecord]
}

extension WorkoutSessionRecord {
    init?(from data: [String: Any]) {
        guard let userId = data["userId"] as? String,
              let setTitle = data["setTitle"] as? String,
              let totalTimeSeconds = data["totalTimeSeconds"] as? Int,
              let estimatedCalories = data["estimatedCalories"] as? Int else { return nil }
        let completedAtSeconds: Double = (data["completedAtSeconds"] as? Double) ?? (data["completedAtSeconds"] as? Int).map { Double($0) } ?? 0
        let exercises: [ExerciseRecord] = (data["exercises"] as? [[String: Any]])?.compactMap { ExerciseRecord(from: $0) } ?? []
        self.init(userId: userId, setTitle: setTitle, completedAtSeconds: completedAtSeconds, totalTimeSeconds: totalTimeSeconds, estimatedCalories: estimatedCalories, exercises: exercises)
    }
}

extension ExerciseRecord {
    init?(from data: [String: Any]) {
        guard let name = data["name"] as? String, let type = data["type"] as? String else { return nil }
        let errors: [ErrorCountRecord] = (data["errors"] as? [[String: Any]])?
            .compactMap { e in (e["reason"] as? String).flatMap { r in (e["count"] as? Int).map { ErrorCountRecord(reason: r, count: $0) } } } ?? []
        self.name = name
        self.type = type
        self.totalReps = data["totalReps"] as? Int
        self.correctReps = data["correctReps"] as? Int
        self.incorrectReps = data["incorrectReps"] as? Int
        self.targetCorrectReps = data["targetCorrectReps"] as? Int
        self.durationSeconds = data["durationSeconds"] as? Double
        self.targetSeconds = data["targetSeconds"] as? Double
        self.errors = errors
    }
}

/// One row for Progress History (from Firestore).
struct WorkoutHistoryItem: Identifiable {
    let id: String
    let setTitle: String
    let completedAt: Date
    let totalTimeSeconds: Int
    let estimatedCalories: Int
    let record: WorkoutSessionRecord

    var formattedDate: String {
        let f = DateFormatter()
        f.dateStyle = .medium
        f.timeStyle = .short
        return f.string(from: completedAt)
    }

    var relativeDate: String {
        let cal = Calendar.current
        if cal.isDateInToday(completedAt) { return "Today" }
        if cal.isDateInYesterday(completedAt) { return "Yesterday" }
        let f = DateFormatter()
        f.dateFormat = "EEEE, MMM d"
        return f.string(from: completedAt)
    }

    var formattedDuration: String { "\(totalTimeSeconds / 60) min \(totalTimeSeconds % 60) s" }
}

extension SessionSummary {
    /// Convert to Firestore-serializable record (caller provides userId and setTitle).
    func toRecord(userId: String, setTitle: String) -> WorkoutSessionRecord {
        let exercises: [ExerciseRecord] = items.map { item in
            switch item {
            case .movement(let name, let totalReps, let correctReps, let incorrectReps, let targetCorrectReps, let errors):
                return ExerciseRecord(
                    name: name,
                    type: "movement",
                    totalReps: totalReps,
                    correctReps: correctReps,
                    incorrectReps: incorrectReps,
                    targetCorrectReps: targetCorrectReps,
                    durationSeconds: nil,
                    targetSeconds: nil,
                    errors: errors.map { ErrorCountRecord(reason: $0.reason, count: $0.count) }
                )
            case .isometric(let name, let durationSeconds, let targetSeconds, let errors):
                return ExerciseRecord(
                    name: name,
                    type: "isometric",
                    totalReps: nil,
                    correctReps: nil,
                    incorrectReps: nil,
                    targetCorrectReps: nil,
                    durationSeconds: durationSeconds,
                    targetSeconds: targetSeconds,
                    errors: errors.map { ErrorCountRecord(reason: $0.reason, count: $0.count) }
                )
            }
        }
        return WorkoutSessionRecord(
            userId: userId,
            setTitle: setTitle,
            completedAtSeconds: Date().timeIntervalSince1970,
            totalTimeSeconds: totalTimeSeconds,
            estimatedCalories: estimatedCalories,
            exercises: exercises
        )
    }
}

