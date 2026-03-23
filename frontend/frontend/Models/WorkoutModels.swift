import Foundation

// MARK: - Date Formatters

/// Thread-safe, lazily-initialized date formatters shared across workout models.
private enum WorkoutDateFormatters {
    static let historyDate: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        return formatter
    }()

    static let relativeHistoryDate: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "EEEE, MMM d"
        return formatter
    }()
}

// MARK: - Workout Session Mode

/// Identifies the type of exercise being performed in a workout session.
enum WorkoutSessionMode: String {
    case wallSit
    case squat
    case plank
    case lunges

    /// Human-readable display name for UI presentation.
    var displayName: String {
        switch self {
        case .wallSit: return "Wall-Sit"
        case .squat:   return "Squat"
        case .plank:   return "Plank"
        case .lunges:  return "Lunges"
        }
    }

    /// Spoken instruction text read aloud via TTS when the exercise begins.
    var demoInstructionText: String {
        switch self {
        case .wallSit:
            return "Wall-sit. Stand sideways to the camera and make sure your full body is visible. "
                + "Keep your back against the wall, lower down until your knees are about 90 degrees. "
                + "Keep your body straight and hold the position."
        case .squat:
            return "Squat. Face the camera and make sure your full body is visible. "
                + "Slowly lower your body until your hips are at knee level. "
                + "Then push back up to a standing position."
        case .plank:
            return "Plank. Stand sideways to the camera and make sure your full body is visible. "
                + "Lower yourself down onto your forearms, keeping your elbows under your shoulders. "
                + "Keep your body straight and hold the position."
        case .lunges:
            return "Lunges. Stand sideways to the camera and make sure your full body is visible. "
                + "Step forward, lower until both knees bend comfortably, "
                + "then push back to the starting position."
        }
    }
}

// MARK: - Workout Program Definition

/// Defines a workout program consisting of one or more exercises.
struct WorkoutProgramDefinition: Identifiable {
    let id = UUID()
    let title: String
    let description: String
    let imageName: String
    let exercises: [WorkoutExercise]
    let initialMode: WorkoutSessionMode
}

// MARK: - Workout Catalog

/// Static catalog of all available workout programs.
enum WorkoutCatalog {
    static let beginnerLevel1 = WorkoutProgramDefinition(
        title: "Beginner Level 1",
        description: "Squat, High Knees, Mountain Climbers",
        imageName: "set1",
        exercises: [
            WorkoutExercise(name: "Wall-Sit", imageName: "wallsit", reps: "5s hold"),
            WorkoutExercise(name: "Squat", imageName: "squat", reps: "3 correct reps"),
            WorkoutExercise(name: "Plank", imageName: "plank", reps: "5s hold")
        ],
        initialMode: .wallSit
    )

    static let beginnerLevel2 = WorkoutProgramDefinition(
        title: "Beginner Level 2",
        description: "Plank",
        imageName: "set2",
        exercises: [
            WorkoutExercise(name: "Plank", imageName: "plank", reps: "5s hold")
        ],
        initialMode: .plank
    )

    static let beginnerLevel3 = WorkoutProgramDefinition(
        title: "Beginner Level 3",
        description: "Lunges",
        imageName: "set3",
        exercises: [
            WorkoutExercise(name: "Lunges", imageName: "lunges", reps: "3 correct reps")
        ],
        initialMode: .lunges
    )

    static let programs: [WorkoutProgramDefinition] = [
        beginnerLevel1,
        beginnerLevel2,
        beginnerLevel3
    ]

    /// Looks up a program by title; falls back to `beginnerLevel1` if not found.
    static func program(for title: String) -> WorkoutProgramDefinition {
        programs.first(where: { $0.title == title }) ?? beginnerLevel1
    }
}

// MARK: - Text Formatting Utilities

/// Shared text formatting helpers for workout durations and dates.
enum WorkoutTextFormatter {
    /// Formats total seconds into "X min Y sec" string.
    static func minuteSecondString(
        for totalSeconds: Int,
        secondSuffix: String = "sec"
    ) -> String {
        "\(totalSeconds / 60) min \(totalSeconds % 60) \(secondSuffix)"
    }

    /// Formats a date using medium date and short time style.
    static func historyDateString(for date: Date) -> String {
        WorkoutDateFormatters.historyDate.string(from: date)
    }

    /// Returns "Today", "Yesterday", or a weekday/date string for the given date.
    static func relativeHistoryDateString(for date: Date) -> String {
        let calendar = Calendar.current
        if calendar.isDateInToday(date) { return "Today" }
        if calendar.isDateInYesterday(date) { return "Yesterday" }
        return WorkoutDateFormatters.relativeHistoryDate.string(from: date)
    }
}

// MARK: - Session Video Storage

/// Resolves local file locations for session recordings saved on this device.
enum SessionVideoStore {
    private static let folderName = "SessionVideos"

    /// Returns the folder used to store recorded workout sessions.
    static func recordingsDirectory() -> URL? {
        guard let baseDirectory = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first else {
            return nil
        }

        let directory = baseDirectory.appendingPathComponent(folderName, isDirectory: true)

        if !FileManager.default.fileExists(atPath: directory.path) {
            do {
                try FileManager.default.createDirectory(
                    at: directory,
                    withIntermediateDirectories: true,
                    attributes: nil
                )
            } catch {
                print("[SessionVideoStore] Create directory error: \(error)")
                return nil
            }
        }

        return directory
    }

    /// Creates a new unique output URL for an upcoming session recording.
    static func makeRecordingURL() -> URL? {
        recordingsDirectory()?.appendingPathComponent("session-\(UUID().uuidString).mov")
    }

    /// Resolves a saved file name into a local file URL if the file still exists.
    static func recordingURL(for fileName: String?) -> URL? {
        guard let fileName,
              let directory = recordingsDirectory() else { return nil }

        let url = directory.appendingPathComponent(fileName)
        return FileManager.default.fileExists(atPath: url.path) ? url : nil
    }

    /// Deletes a previously recorded local session file, if present.
    static func deleteRecording(fileName: String?) {
        guard let url = recordingURL(for: fileName) else { return }

        do {
            try FileManager.default.removeItem(at: url)
        } catch {
            print("[SessionVideoStore] Delete error: \(error)")
        }
    }
}

// MARK: - Progress / Summary Display Models

/// Represents daily calorie data for the weekly chart.
struct DailyCalorie: Identifiable {
    let id = UUID()
    let date: Date
    let calories: Int
}

/// Aggregate summary of one workout set (e.g., "Beginner Level 1 — completed 3 times").
struct WorkoutSetSummary: Identifiable {
    let id = UUID()
    let name: String
    let timesCompleted: Int
    let totalCalories: Int
}

// MARK: - Workout Exercise

/// A single exercise within a workout program (used in the detail/preview screen).
struct WorkoutExercise: Identifiable {
    let id = UUID()
    let name: String
    let imageName: String
    let reps: String
}

// MARK: - Workout Result Summary

/// A single type of error with its occurrence count.
struct ErrorCount: Identifiable {
    let id = UUID()
    let reason: String
    let count: Int
}

/// A timestamped mistake event during a workout (optionally tied to a rep number).
struct MistakeEvent: Identifiable {
    let id = UUID()
    let atSecond: Double
    let reason: String
    let repNumber: Int?
}

/// Summary for one exercise within a session result.
/// `.movement` is for rep-based exercises (squat, lunges);
/// `.isometric` is for hold-based exercises (wall-sit, plank).
enum ExerciseSummaryItem: Identifiable {
    case movement(
        name: String,
        totalReps: Int,
        correctReps: Int,
        incorrectReps: Int,
        targetCorrectReps: Int,
        errors: [ErrorCount],
        mistakes: [MistakeEvent]
    )
    case isometric(
        name: String,
        durationSeconds: Double,
        targetSeconds: Double,
        errors: [ErrorCount],
        mistakes: [MistakeEvent]
    )

    var id: String {
        switch self {
        case .movement(let name, _, _, _, _, _, _): return "movement-\(name)"
        case .isometric(let name, _, _, _, _): return "isometric-\(name)"
        }
    }

    var displayName: String {
        switch self {
        case .movement(let name, _, _, _, _, _, _): return name
        case .isometric(let name, _, _, _, _): return name
        }
    }
}

/// Complete summary of a workout session, including all exercise items and totals.
struct SessionSummary {
    let items: [ExerciseSummaryItem]
    let totalTimeSeconds: Int
    let estimatedCalories: Int
    let sessionVideoFileName: String?
}

// MARK: - Firestore Workout History (Codable for persistence)

/// Codable error record for Firestore serialization.
struct ErrorCountRecord: Codable {
    let reason: String
    let count: Int
}

/// Codable mistake event record for Firestore serialization.
struct MistakeEventRecord: Codable {
    let atSecond: Double
    let reason: String
    let repNumber: Int?
}

/// Codable record for one exercise within a saved session.
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
    let mistakes: [MistakeEventRecord]
}

/// Codable record for an entire workout session saved to Firestore.
struct WorkoutSessionRecord: Codable {
    let userId: String
    let setTitle: String
    let completedAtSeconds: Double
    let totalTimeSeconds: Int
    let estimatedCalories: Int
    let exercises: [ExerciseRecord]
    let sessionVideoFileName: String?
}

// MARK: - WorkoutSessionRecord Firestore Deserialization

extension WorkoutSessionRecord {
    /// Failable initializer that parses a raw Firestore document dictionary.
    init?(from data: [String: Any]) {
        guard let userId = data["userId"] as? String,
              let setTitle = data["setTitle"] as? String,
              let totalTimeSeconds = data["totalTimeSeconds"] as? Int,
              let estimatedCalories = data["estimatedCalories"] as? Int else { return nil }
        let completedAtSeconds: Double =
            (data["completedAtSeconds"] as? Double)
            ?? (data["completedAtSeconds"] as? Int).map(Double.init)
            ?? 0
        let exercises: [ExerciseRecord] = (data["exercises"] as? [[String: Any]])?.compactMap { ExerciseRecord(from: $0) } ?? []
        let sessionVideoFileName = data["sessionVideoFileName"] as? String
        self.init(
            userId: userId,
            setTitle: setTitle,
            completedAtSeconds: completedAtSeconds,
            totalTimeSeconds: totalTimeSeconds,
            estimatedCalories: estimatedCalories,
            exercises: exercises,
            sessionVideoFileName: sessionVideoFileName
        )
    }
}

// MARK: - ExerciseRecord Firestore Deserialization

extension ExerciseRecord {
    /// Failable initializer that parses a raw Firestore exercise dictionary.
    init?(from data: [String: Any]) {
        guard let name = data["name"] as? String,
              let type = data["type"] as? String else { return nil }
        let errors: [ErrorCountRecord] = (data["errors"] as? [[String: Any]])?
            .compactMap { entry in
                (entry["reason"] as? String).flatMap { reason in
                    (entry["count"] as? Int).map { count in
                        ErrorCountRecord(reason: reason, count: count)
                    }
                }
            } ?? []
        let mistakes: [MistakeEventRecord] = (data["mistakes"] as? [[String: Any]])?
            .compactMap { m in
                (m["reason"] as? String).flatMap { reason in
                    ((m["atSecond"] as? Double) ?? (m["atSecond"] as? Int).map(Double.init)).map { at in
                        let repNumber = (m["repNumber"] as? Int) ?? (m["repNumber"] as? Double).map { Int($0) }
                        return MistakeEventRecord(atSecond: at, reason: reason, repNumber: repNumber)
                    }
                }
            } ?? []
        self.name = name
        self.type = type
        self.totalReps = data["totalReps"] as? Int
        self.correctReps = data["correctReps"] as? Int
        self.incorrectReps = data["incorrectReps"] as? Int
        self.targetCorrectReps = data["targetCorrectReps"] as? Int
        self.durationSeconds = data["durationSeconds"] as? Double
        self.targetSeconds = data["targetSeconds"] as? Double
        self.errors = errors
        self.mistakes = mistakes
    }
}

// MARK: - Workout History Item

/// One row in the Progress History list, combining a Firestore document ID with display data.
struct WorkoutHistoryItem: Identifiable {
    let id: String
    let setTitle: String
    let completedAt: Date
    let totalTimeSeconds: Int
    let estimatedCalories: Int
    let record: WorkoutSessionRecord

    var formattedDate: String {
        WorkoutTextFormatter.historyDateString(for: completedAt)
    }

    var relativeDate: String {
        WorkoutTextFormatter.relativeHistoryDateString(for: completedAt)
    }

    var formattedDuration: String {
        WorkoutTextFormatter.minuteSecondString(
            for: totalTimeSeconds,
            secondSuffix: "s"
        )
    }
}

// MARK: - SessionSummary to Firestore Conversion

extension SessionSummary {
    /// Converts this summary to a Firestore-serializable record.
    func toRecord(userId: String, setTitle: String) -> WorkoutSessionRecord {
        let exercises: [ExerciseRecord] = items.map { item in
            switch item {
            case .movement(let name, let totalReps, let correctReps, let incorrectReps, let targetCorrectReps, let errors, let mistakes):
                return ExerciseRecord(
                    name: name,
                    type: "movement",
                    totalReps: totalReps,
                    correctReps: correctReps,
                    incorrectReps: incorrectReps,
                    targetCorrectReps: targetCorrectReps,
                    durationSeconds: nil,
                    targetSeconds: nil,
                    errors: errors.map { ErrorCountRecord(reason: $0.reason, count: $0.count) },
                    mistakes: mistakes.map { MistakeEventRecord(atSecond: $0.atSecond, reason: $0.reason, repNumber: $0.repNumber) }
                )
            case .isometric(let name, let durationSeconds, let targetSeconds, let errors, let mistakes):
                return ExerciseRecord(
                    name: name,
                    type: "isometric",
                    totalReps: nil,
                    correctReps: nil,
                    incorrectReps: nil,
                    targetCorrectReps: nil,
                    durationSeconds: durationSeconds,
                    targetSeconds: targetSeconds,
                    errors: errors.map { ErrorCountRecord(reason: $0.reason, count: $0.count) },
                    mistakes: mistakes.map { MistakeEventRecord(atSecond: $0.atSecond, reason: $0.reason, repNumber: $0.repNumber) }
                )
            }
        }
        return WorkoutSessionRecord(
            userId: userId,
            setTitle: setTitle,
            completedAtSeconds: Date().timeIntervalSince1970,
            totalTimeSeconds: totalTimeSeconds,
            estimatedCalories: estimatedCalories,
            exercises: exercises,
            sessionVideoFileName: sessionVideoFileName
        )
    }
}
