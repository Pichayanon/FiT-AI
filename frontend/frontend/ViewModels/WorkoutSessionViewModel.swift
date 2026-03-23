import Foundation
import SwiftUI

@MainActor
final class WorkoutSessionViewModel: ObservableObject {
    enum BackendPhase: String {
        case noPose = "NO_POSE"
        case havePose = "HAVE_POSE"
        case buffering = "BUFFERING"
        case inferencing = "INFERENCING"
    }

    @Published var cameraManager = CameraService()
    let speech = SpeechService()
    let workoutHistory = WorkoutHistoryService()

    let setTitle: String
    private let program: WorkoutProgramDefinition

    @Published var feedback: String = "Press Start when Ready"
    @Published var totalReps = 0
    @Published var correctReps = 0
    @Published var incorrectReps = 0
    @Published var startTime = Date()
    @Published var navigateToResult = false
    @Published var isSessionRunning = false
    @Published var isFinalizingSession = false
    @Published var backendState: String = BackendPhase.noPose.rawValue

    @Published var squatStandOK: Bool = false
    @Published var squatStarted: Bool = false
    @Published var lastStandOKAt: Date?
    @Published var lastPleaseStartAt: Date = .distantPast
    @Published var lastSpokenCorrectReps: Int = 0

    var activeRepeatableSquatStandIssue: String?
    var activeRepeatableSquatStandIssueSince: Date?
    var lastRepeatableSquatStandIssueSpokenAt: Date = .distantPast
    var wallSitNeedsStartSince: Date?
    var lastWallSitStartReminderAt: Date = .distantPast
    var plankNeedsStartSince: Date?
    var lastPlankStartReminderAt: Date = .distantPast
    var lungesNeedsStartSince: Date?
    var lastLungesStartReminderAt: Date = .distantPast

    @Published var wallSitHold = IsometricHoldState(
        holdTargetSeconds: 5.0,
        targetSeconds: 5.0
    )
    @Published var passedWallSit: Bool = false

    @Published var plankHold = IsometricHoldState()
    @Published var passedPlank: Bool = false

    @Published var showExerciseIntroOverlay: Bool = false
    @Published var showSquatPreview: Bool = false
    @Published var squatPreviewSeconds: Int = 8
    @Published var showPlankPreview: Bool = false
    @Published var plankPreviewSeconds: Int = 8

    @Published var didSwitchToSquat: Bool = false
    @Published var didSwitchToPlank: Bool = false

    @Published var squatTargetCorrectReps: Int = 3
    @Published var lungesTargetCorrectReps: Int = 3

    @Published var mode: WorkoutSessionMode = .wallSit

    @Published var sessionSummary: SessionSummary = SessionSummary(
        items: [],
        totalTimeSeconds: 0,
        estimatedCalories: 0,
        sessionVideoFileName: nil
    )

    var wallSitErrors = ExerciseErrorTracker()
    var squatErrors = ExerciseErrorTracker()
    var plankErrors = ExerciseErrorTracker()
    var lungesErrors = ExerciseErrorTracker()

    let speechLang = "en-US"
    let previewDurationSeconds = 8
    let introOverlayPostSpeechDelaySeconds = 2.0
    let isSessionRecordingEnabled = true

    var initialMode: WorkoutSessionMode {
        program.initialMode
    }

    var activeWSURL: String {
        AppConfig.webSocketURL(for: mode)
    }

    let showTestVoiceButton: Bool = false

    var wallSitProgress01: Double { wallSitHold.progress }

    var squatProgress01: Double {
        let target = Double(max(1, squatTargetCorrectReps))
        return min(1.0, max(0.0, Double(correctReps) / target))
    }

    var plankProgress01: Double { plankHold.progress }

    var lungesProgress01: Double {
        let target = Double(max(1, lungesTargetCorrectReps))
        return min(1.0, max(0.0, Double(correctReps) / target))
    }

    var correctSeconds: Double { wallSitHold.elapsedSeconds }
    var targetSeconds: Double { wallSitHold.targetSeconds }
    var wallSitConsecutiveCorrect: Int { wallSitHold.consecutiveCorrectCount }
    var wallSitCountingActive: Bool { wallSitHold.isCountingActive }

    var plankCorrectSeconds: Double { plankHold.elapsedSeconds }
    var plankTargetSeconds: Double { plankHold.targetSeconds }

    var titleForHUD: String {
        setTitle
    }

    var sessionGuidanceOverlayText: String? {
        guard isSessionRunning else { return nil }
        let isMovementWaitingState =
            (mode == .squat || mode == .lunges) && backendState == "WAITING"
        let isPlankPostureSetupState =
            mode == .plank
            && backendState == BackendPhase.havePose.rawValue
            && feedback.trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased()
                .contains("plank position")
        let isSetupIssueState =
            backendState == BackendPhase.noPose.rawValue
            || isMovementWaitingState
            || isPlankPostureSetupState
        guard isSetupIssueState else { return nil }

        let normalizedFeedback =
            feedback.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if normalizedFeedback.contains("adjust your lights") {
            return "Please adjust your lights."
        }
        if normalizedFeedback.contains("adjust your camera") {
            return makeFullBodyGuidanceText()
        }
        if normalizedFeedback.contains("full body") {
            return makeFullBodyGuidanceText()
        }
        if normalizedFeedback.contains("plank position") {
            return "Lower into plank position"
        }
        return nil
    }

    init(setTitle: String) {
        let program = WorkoutCatalog.program(for: setTitle)
        self.program = program
        self.setTitle = program.title
        self.mode = program.initialMode
    }
}
