import Foundation
import SwiftUI

// MARK: - Exercise Error Tracker

/// Tracks per-exercise error counts and mistake events with deduplication.
///
/// Only records a new error when the prediction transitions from correct to incorrect,
/// preventing the same sustained error from being counted on every backend message.
struct ExerciseErrorTracker {
    var errorCounts: [String: Int] = [:]
    var mistakeEvents: [MistakeEvent] = []
    var lastPredictionWasCorrect: Bool = true

    /// Records an error if transitioning from correct to incorrect.
    /// Only increments the count on the first incorrect prediction after a correct one.
    mutating func recordMistakeIfTransitioned(
        prediction: String,
        elapsedSeconds: Int,
        repNumber: Int?,
        displayLabel: String
    ) {
        if lastPredictionWasCorrect {
            errorCounts[prediction] = (errorCounts[prediction] ?? 0) + 1
            mistakeEvents.append(
                MistakeEvent(
                    atSecond: elapsedSeconds,
                    reason: displayLabel,
                    repNumber: repNumber
                )
            )
        }
        lastPredictionWasCorrect = false
    }

    /// Marks the current state as correct (resets the transition guard).
    mutating func markPredictionAsCorrect() {
        lastPredictionWasCorrect = true
    }

    /// Resets all tracking state.
    mutating func reset() {
        errorCounts = [:]
        mistakeEvents = []
        lastPredictionWasCorrect = true
    }
}

// MARK: - Isometric Hold State

/// Shared state machine for isometric hold exercises (wall-sit, plank).
///
/// Two-phase progression:
/// 1. Gate phase — accumulate 3 consecutive correct predictions to prove stable form.
/// 2. Hold phase — count elapsed hold time until the target is reached.
///
/// Dropping out of correct form at any point resets the entire progression.
struct IsometricHoldState {
    var consecutiveCorrectCount: Int = 0
    var isCountingActive: Bool = false
    var isCurrentlyCorrect: Bool = false
    var elapsedSeconds: Double = 0
    var targetSeconds: Double = 5.0
    var hasPassed: Bool = false

    /// Progress from 0 to 1 for UI display.
    var progress: Double {
        guard targetSeconds > 0 else { return 0 }
        return min(1.0, max(0.0, elapsedSeconds / targetSeconds))
    }

    /// Called by a 0.1s timer tick while hold is active.
    /// Returns true if the target was just reached.
    mutating func updateHoldTime(interval: Double) -> Bool {
        guard isCountingActive, !hasPassed, isCurrentlyCorrect else { return false }
        elapsedSeconds = min(targetSeconds, elapsedSeconds + interval)
        if elapsedSeconds >= targetSeconds {
            elapsedSeconds = targetSeconds
            hasPassed = true
            return true
        }
        return false
    }

    /// Possible actions returned by `handleResult`.
    enum Action {
        case none
        case firstCorrectInStreak
        case gatePassed(feedbackText: String)
        case holdingProgress(feedbackText: String)
        case lostForm(feedbackText: String)
    }

    /// Processes a backend prediction result and returns the appropriate UI action.
    mutating func processIsometricPrediction(isCorrect: Bool, incorrectFeedback: String, exerciseName: String) -> Action {
        isCurrentlyCorrect = isCorrect

        if hasPassed { return .none }

        // Phase 1: Gate — require 3 consecutive correct predictions
        if !isCountingActive {
            if isCorrect {
                consecutiveCorrectCount = min(3, consecutiveCorrectCount + 1)
                if consecutiveCorrectCount == 1 {
                    return .firstCorrectInStreak
                }
                if consecutiveCorrectCount >= 3 {
                    isCountingActive = true
                    elapsedSeconds = 0
                    targetSeconds = 5.0
                    return .gatePassed(feedbackText: "Start hold...")
                }
                return .holdingProgress(feedbackText: "Correct \(consecutiveCorrectCount)/3 - Keep holding")
            } else {
                if consecutiveCorrectCount != 0 {
                    consecutiveCorrectCount = 0
                    return .lostForm(feedbackText: incorrectFeedback)
                }
                return .none
            }
        }

        // Phase 2: Hold — counting down the target seconds
        if !isCorrect {
            isCountingActive = false
            isCurrentlyCorrect = false
            consecutiveCorrectCount = 0
            elapsedSeconds = 0
            return .lostForm(feedbackText: incorrectFeedback)
        }

        return .none
    }

    /// Resets all hold state to initial values.
    mutating func reset() {
        consecutiveCorrectCount = 0
        isCountingActive = false
        isCurrentlyCorrect = false
        elapsedSeconds = 0
        targetSeconds = 5.0
        hasPassed = false
    }
}

// MARK: - Workout Session View Model

/// Coordinates workout session state, camera/backend messaging, speech feedback,
/// exercise progression, and session summary generation.
///
/// Supports four exercise types across two categories:
/// - Isometric (wall-sit, plank): 3-correct gate then timed hold
/// - Movement (squat, lunges): rep counting with error tracking
///
/// Multi-exercise programs progress automatically (e.g., wall-sit -> squat -> plank)
/// with preview countdowns between transitions.
@MainActor
final class WorkoutSessionViewModel: ObservableObject {

    /// Backend processing phases (matched to Python server).
    private enum BackendPhase: String {
        case NO_POSE, HAVE_POSE, BUFFERING, INFERENCING
    }

    // MARK: - Dependencies

    @Published var cameraManager = CameraService()
    let speech = SpeechService()
    private let workoutHistory = WorkoutHistoryService()

    // MARK: - Inputs

    let setTitle: String
    private let program: WorkoutProgramDefinition

    // MARK: - Published UI State

    @Published var feedback: String = "Side view - Stand -> Lower -> Hold"
    @Published var totalReps = 0
    @Published var correctReps = 0
    @Published var incorrectReps = 0
    @Published var startTime = Date()
    @Published var navigateToResult = false
    @Published var isSessionRunning = false
    @Published var backendState: String = BackendPhase.NO_POSE.rawValue

    // Squat speech control
    @Published var squatStandOK: Bool = false
    @Published var squatStarted: Bool = false
    @Published var lastStandOKAt: Date?
    @Published var lastPleaseStartAt: Date = .distantPast
    @Published var lastSpokenCorrectReps: Int = 0

    // Isometric hold state (wall-sit)
    @Published var wallSitHold = IsometricHoldState()
    @Published var passedWallSit: Bool = false

    // Isometric hold state (plank)
    @Published var plankHold = IsometricHoldState()
    @Published var passedPlank: Bool = false

    // Exercise preview overlays
    @Published var showSquatPreview: Bool = false
    @Published var squatPreviewSeconds: Int = 8
    @Published var showPlankPreview: Bool = false
    @Published var plankPreviewSeconds: Int = 8

    // Auto-switch guards (prevent double-switching)
    @Published var didSwitchToSquat: Bool = false
    @Published var didSwitchToPlank: Bool = false

    // Target reps for movement exercises
    @Published var squatTargetCorrectReps: Int = 3
    @Published var lungesTargetCorrectReps: Int = 3

    // Current exercise mode
    @Published var mode: WorkoutSessionMode = .wallSit

    /// Session summary populated when the session finishes; drives the result screen.
    @Published var sessionSummary: SessionSummary = SessionSummary(items: [], totalTimeSeconds: 0, estimatedCalories: 0)

    // MARK: - Error Trackers (per-exercise)

    private var wallSitErrors = ExerciseErrorTracker()
    private var squatErrors = ExerciseErrorTracker()
    private var plankErrors = ExerciseErrorTracker()
    private var lungesErrors = ExerciseErrorTracker()

    // MARK: - Constants

    private let speechLang = "en-US"
    private let previewDurationSeconds = 8

    // WebSocket endpoints (one per exercise backend)
    private let wsWallSitURL = "ws://172.20.10.5:5050/ws/video"
    private let wsSquatURL   = "ws://172.20.10.5:5051/ws/video"
    private let wsPlankURL   = "ws://172.20.10.5:5052/ws/video"
    private let wsLungesURL  = "ws://172.20.10.5:5053/ws/video"

    private var initialMode: WorkoutSessionMode {
        program.initialMode
    }

    /// Returns the WebSocket URL for the currently active exercise mode.
    private var activeWSURL: String {
        switch mode {
        case .wallSit: return wsWallSitURL
        case .squat:   return wsSquatURL
        case .plank:   return wsPlankURL
        case .lunges:  return wsLungesURL
        }
    }

    // Confidence thresholds for isometric classification
    private let wallSitConfThreshold: Double = 0.50
    private let plankConfThreshold: Double = 0.50

    // Debug: toggle the test voice button in the UI
    let showTestVoiceButton: Bool = false

    // MARK: - Computed Properties

    /// Wall-sit hold progress (0...1) for the progress bar.
    var wallSitProgress01: Double { wallSitHold.progress }

    /// Squat rep progress (0...1) for the progress bar.
    var squatProgress01: Double {
        let tgt = Double(max(1, squatTargetCorrectReps))
        return min(1.0, max(0.0, Double(correctReps) / tgt))
    }

    /// Plank hold progress (0...1) for the progress bar.
    var plankProgress01: Double { plankHold.progress }

    /// Lunges rep progress (0...1) for the progress bar.
    var lungesProgress01: Double {
        let tgt = Double(max(1, lungesTargetCorrectReps))
        return min(1.0, max(0.0, Double(correctReps) / tgt))
    }

    /// Convenience accessors so the View can read hold state without reaching into the struct.
    var correctSeconds: Double { wallSitHold.elapsedSeconds }
    var targetSeconds: Double { wallSitHold.targetSeconds }
    var wallSitConsecutiveCorrect: Int { wallSitHold.consecutiveCorrectCount }
    var wallSitCountingActive: Bool { wallSitHold.isCountingActive }

    var plankCorrectSeconds: Double { plankHold.elapsedSeconds }
    var plankTargetSeconds: Double { plankHold.targetSeconds }

    /// Title shown in the top HUD (set name + current exercise).
    var titleForHUD: String {
        "\(setTitle) - \(mode.displayName)"
    }

    // MARK: - Initialization

    init(setTitle: String) {
        let program = WorkoutCatalog.program(for: setTitle)
        self.program = program
        self.setTitle = program.title
        self.mode = program.initialMode
    }

    // MARK: - View Lifecycle

    /// Called when the session view appears. Starts camera and registers backend message handler.
    func onAppear() {
        speech.configureAudioSession()
        cameraManager.startSession()
        cameraManager.onBackendMessage = { [weak self] text in
            self?.processBackendMessage(text)
        }
    }

    /// Called when the session view disappears. Stops session and camera.
    func onDisappear() {
        stopSession()
        cameraManager.stopSession()
    }

    // MARK: - Session Control

    /// Starts the workout session: connects to the backend, resets counters, begins streaming.
    func startSession() {
        withAnimation(.easeInOut(duration: 0.2)) {
            isSessionRunning = true
        }

        startTime = Date()
        backendState = BackendPhase.NO_POSE.rawValue
        feedback = "Streaming to backend..."

        resetSessionState()

        cameraManager.startStreaming(to: activeWSURL)
        speech.speak("Session started", language: speechLang, minInterval: 0)
        scheduleDemoInstructionIfNeeded(for: mode)
    }

    /// Stops the workout session: disconnects streaming, resets backend state, stops speech.
    func stopSession() {
        isSessionRunning = false
        cameraManager.stopStreaming()
        backendState = BackendPhase.NO_POSE.rawValue
        speech.stop()
    }

    /// Resets all session-scoped counters and state before connecting to a workout backend.
    private func resetSessionState() {
        mode = initialMode

        didSwitchToSquat = false
        didSwitchToPlank = false
        passedWallSit = false

        showSquatPreview = false
        squatPreviewSeconds = previewDurationSeconds
        showPlankPreview = false
        plankPreviewSeconds = previewDurationSeconds

        totalReps = 0
        correctReps = 0
        incorrectReps = 0
        lastSpokenCorrectReps = 0

        wallSitHold.reset()
        plankHold.reset()
        passedPlank = false

        squatStandOK = false
        squatStarted = false
        lastStandOKAt = nil
        lastPleaseStartAt = .distantPast

        wallSitErrors.reset()
        squatErrors.reset()
        plankErrors.reset()
        lungesErrors.reset()
    }

    /// Speaks the exercise setup instructions once after the stream begins.
    /// Skipped for squat since it has its own stand-check flow.
    private func scheduleDemoInstructionIfNeeded(for mode: WorkoutSessionMode) {
        guard mode != .squat else { return }

        DispatchQueue.main.asyncAfter(deadline: .now() + 1.0) { [weak self] in
            guard let self else { return }
            guard self.isSessionRunning else { return }

            self.speech.speak(
                mode.demoInstructionText,
                language: self.speechLang,
                minInterval: 0,
                allowRepeat: true
            )
        }
    }

    /// Finishes the workout session: stops streaming, builds summary, saves to history, navigates to results.
    func finishSession() {
        stopSession()
        speech.speak("Session complete", language: speechLang, minInterval: 0)
        sessionSummary = buildSessionSummary()
        Task { await workoutHistory.saveSession(summary: sessionSummary, setTitle: setTitle) }
        navigateToResult = true
    }

    // MARK: - Timer Handlers

    /// Called every 0.1s by the wall-sit timer. Increments hold time if conditions are met.
    func handleWallSitTick() {
        guard isSessionRunning, mode == .wallSit, !passedWallSit, !showSquatPreview else { return }
        let justPassed = wallSitHold.updateHoldTime(interval: 0.1)
        if justPassed {
            passedWallSit = true
            setFeedbackIfChanged("Passed")
            speech.speak("Wall-sit complete. Switch to Squat.", language: speechLang, minInterval: 0)
            // Stop streaming during preview to prevent stale detections
            cameraManager.stopStreaming()
            beginTransitionCountdown(for: .squat)
        }
    }

    /// Called every 0.5s to check if the user has been idle too long after squat stand OK.
    func handleSquatIdleTick() {
        guard isSessionRunning, mode == .squat, squatStandOK, !squatStarted else { return }
        guard let okAt = lastStandOKAt else { return }

        // Remind the user after 30 seconds of inactivity post stand-OK
        if Date().timeIntervalSince(okAt) > 30.0 {
            if Date().timeIntervalSince(lastPleaseStartAt) > 15.0 {
                speech.speak("Please start squat", language: speechLang, minInterval: 0)
                lastPleaseStartAt = Date()
            }
        }
    }

    /// Called every 0.1s by the plank timer. Increments hold time if conditions are met.
    func handlePlankTick() {
        guard isSessionRunning, mode == .plank, !passedPlank else { return }
        let justPassed = plankHold.updateHoldTime(interval: 0.1)
        if justPassed {
            passedPlank = true
            setFeedbackIfChanged("Plank completed")
            finishSession()
        }
    }

    // MARK: - Feedback Speech

    /// Called whenever feedback text changes. Cleans and speaks it if appropriate.
    func handleFeedbackChange() {
        guard isSessionRunning else { return }
        let cleaned = speech.cleanForSpeech(feedback)
        guard speech.shouldSpeak(cleaned) else { return }
        speech.speak(cleaned, language: speechLang, minInterval: 1.2)
    }

    // MARK: - Exercise Preview and Switching

    /// Starts a countdown preview overlay before transitioning to the next exercise.
    /// Used for both squat preview (after wall-sit) and plank preview (after squat).
    private func beginTransitionCountdown(for nextMode: WorkoutSessionMode) {
        let isSquat = (nextMode == .squat)

        if isSquat {
            showSquatPreview = true
            squatPreviewSeconds = previewDurationSeconds
        } else {
            showPlankPreview = true
            plankPreviewSeconds = previewDurationSeconds
        }

        speech.speak(
            nextMode.demoInstructionText,
            language: speechLang,
            minInterval: 0,
            allowRepeat: true
        )

        // Countdown UI updates
        for i in 1...previewDurationSeconds {
            DispatchQueue.main.asyncAfter(deadline: .now() + Double(i)) { [weak self] in
                guard let self else { return }
                let remaining = max(0, self.previewDurationSeconds - i)
                if isSquat {
                    if self.showSquatPreview { self.squatPreviewSeconds = remaining }
                } else {
                    if self.showPlankPreview { self.plankPreviewSeconds = remaining }
                }
            }
        }

        // Auto-switch after countdown completes
        DispatchQueue.main.asyncAfter(deadline: .now() + Double(previewDurationSeconds)) { [weak self] in
            guard let self else { return }
            if isSquat {
                self.showSquatPreview = false
                self.switchToSquat()
            } else {
                self.showPlankPreview = false
                self.switchToPlank()
            }
        }
    }

    /// Transitions to squat mode with fresh counters and a new WebSocket connection.
    private func switchToSquat() {
        guard isSessionRunning, !didSwitchToSquat else { return }

        didSwitchToSquat = true
        mode = .squat
        backendState = BackendPhase.NO_POSE.rawValue
        setFeedbackIfChanged("PASSED (Switch to Squat)")

        // Reset movement counters for the new exercise
        totalReps = 0
        correctReps = 0
        incorrectReps = 0
        lastSpokenCorrectReps = 0

        squatStandOK = false
        squatStarted = false
        lastStandOKAt = nil
        lastPleaseStartAt = .distantPast

        // Stop the wall-sit hold timer from counting further
        wallSitHold.isCountingActive = false
        wallSitHold.isCurrentlyCorrect = false

        cameraManager.stopStreaming()
        cameraManager.startStreaming(to: activeWSURL)
    }

    /// Transitions to plank mode with fresh counters and a new WebSocket connection.
    private func switchToPlank() {
        guard isSessionRunning, !didSwitchToPlank else { return }

        didSwitchToPlank = true
        mode = .plank
        backendState = BackendPhase.NO_POSE.rawValue
        setFeedbackIfChanged("PASSED (Switch to Plank)")

        plankHold.reset()
        passedPlank = false
        plankErrors.reset()

        cameraManager.stopStreaming()
        cameraManager.startStreaming(to: activeWSURL)
    }

    // MARK: - Backend Message Handling

    /// Parses and routes incoming WebSocket messages from the backend.
    private func processBackendMessage(_ text: String) {
        guard let data = text.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return
        }

        let type = obj["type"] as? String ?? ""

        if type == "status" {
            processStatusMessage(obj)
            return
        }

        if type == "info" {
            let msg = obj["message"] as? String ?? "..."
            DispatchQueue.main.async { self.setFeedbackIfChanged(msg) }
            return
        }

        if type == "result" {
            processResultMessage(obj)
            return
        }
    }

    /// Handles backend status messages (pose detection state changes).
    private func processStatusMessage(_ obj: [String: Any]) {
        let stateRaw = (obj["state"] as? String ?? BackendPhase.NO_POSE.rawValue).uppercased()
        let msg = obj["message"] as? String

        DispatchQueue.main.async {
            self.backendState = stateRaw

            // Reset hold state when pose is lost to prevent timers from counting stale data
            if stateRaw == BackendPhase.NO_POSE.rawValue {
                if self.mode == .wallSit {
                    self.wallSitHold.isCurrentlyCorrect = false
                    self.wallSitHold.isCountingActive = false
                    self.wallSitHold.consecutiveCorrectCount = 0
                    self.wallSitHold.elapsedSeconds = 0
                } else if self.mode == .plank {
                    self.plankHold.isCurrentlyCorrect = false
                    self.plankHold.isCountingActive = false
                    self.plankHold.consecutiveCorrectCount = 0
                    self.plankHold.elapsedSeconds = 0
                }
            }

            if let msg { self.setFeedbackIfChanged(msg) }
        }
    }

    /// Routes result messages to the appropriate exercise handler.
    private func processResultMessage(_ obj: [String: Any]) {
        let predRaw = (obj["prediction"] as? String ?? "...")
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        let conf = obj["confidence"] as? Double

        DispatchQueue.main.async {
            // Update feedback display
            if let conf {
                self.setFeedbackIfChanged("\(predRaw) - \(String(format: "%.2f", conf))")
            } else {
                self.setFeedbackIfChanged("\(predRaw)")
            }

            switch self.mode {
            case .wallSit:
                self.processWallSitPrediction(pred: predRaw, conf: conf)
            case .squat:
                self.processSquatPrediction(pred: predRaw, conf: conf, obj: obj)
            case .plank:
                self.processPlankPrediction(pred: predRaw, conf: conf)
            case .lunges:
                self.processLungesPrediction(pred: predRaw, obj: obj)
            }
        }
    }

    // MARK: - Wall-Sit Result Handler

    /// Processes wall-sit predictions using the isometric hold state machine.
    private func processWallSitPrediction(pred: String, conf: Double?) {
        let p = pred.lowercased()
        let c = conf ?? 0.0
        let isCorrectNow = p.contains("correct") && c >= wallSitConfThreshold

        // Track errors with deduplication
        if isCorrectNow {
            wallSitErrors.markPredictionAsCorrect()
        } else if shouldTrackAsMistakeLabel(pred) {
            wallSitErrors.recordMistakeIfTransitioned(
                prediction: pred,
                elapsedSeconds: sessionElapsedSecond(),
                repNumber: nil,
                displayLabel: Self.displayLabel(for: pred)
            )
        }

        if passedWallSit { return }

        let action = wallSitHold.processIsometricPrediction(
            isCorrect: isCorrectNow,
            incorrectFeedback: incorrectFeedbackText(for: pred),
            exerciseName: "Wall-Sit"
        )

        applyIsometricAction(action)
    }

    // MARK: - Plank Result Handler

    /// Processes plank predictions using the isometric hold state machine.
    private func processPlankPrediction(pred: String, conf: Double?) {
        let p = pred.lowercased()
        let c = conf ?? 0.0
        let isCorrectNow = p.contains("correct") && c >= plankConfThreshold

        // Track errors with deduplication
        if isCorrectNow {
            plankErrors.markPredictionAsCorrect()
        } else if shouldTrackAsMistakeLabel(pred) {
            plankErrors.recordMistakeIfTransitioned(
                prediction: pred,
                elapsedSeconds: sessionElapsedSecond(),
                repNumber: nil,
                displayLabel: Self.displayLabel(for: pred)
            )
        }

        if passedPlank { return }

        let action = plankHold.processIsometricPrediction(
            isCorrect: isCorrectNow,
            incorrectFeedback: incorrectFeedbackText(for: pred),
            exerciseName: "Plank"
        )

        applyIsometricAction(action)
    }

    /// Applies a hold state action to the UI (speech and feedback).
    private func applyIsometricAction(_ action: IsometricHoldState.Action) {
        switch action {
        case .none:
            break
        case .firstCorrectInStreak:
            speech.speak("OK", language: speechLang, minInterval: 0)
        case .gatePassed(let text):
            setFeedbackIfChanged(text)
        case .holdingProgress(let text):
            setFeedbackIfChanged(text)
        case .lostForm(let text):
            setFeedbackIfChanged(text)
        }
    }

    // MARK: - Squat Result Handler

    /// Processes squat predictions, handling both stand-check and bottom-position phases.
    private func processSquatPrediction(pred: String, conf: Double?, obj: [String: Any]) {
        let backendMode = obj["mode"] as? String ?? ""

        // Phase 1: Stand check (user must be in correct standing position first)
        if backendMode == "stand" {
            let standOk = obj["stand_ok"] as? Bool ?? false
            if standOk {
                squatErrors.markPredictionAsCorrect()
                if !squatStandOK {
                    squatStandOK = true
                    lastStandOKAt = Date()
                    speech.speak("Stand OK", language: speechLang, minInterval: 0)
                }
            } else {
                speech.speak(Self.displayLabel(for: pred), language: speechLang, minInterval: 2.0)
            }
            return
        }

        // Phase 2: Bottom position (actual squat reps)
        squatStarted = true
        processMovementPrediction(
            pred: pred,
            obj: obj,
            tracker: &squatErrors,
            goodPrefix: "good",
            targetReps: squatTargetCorrectReps,
            onComplete: { [weak self] in
                guard let self else { return }
                self.speech.speak("Squat complete. Switch to Plank.", language: self.speechLang, minInterval: 0)
                self.cameraManager.stopStreaming()
                self.beginTransitionCountdown(for: .plank)
            }
        )
    }

    // MARK: - Lunges Result Handler

    /// Processes lunges predictions using the shared movement rep handler.
    private func processLungesPrediction(pred: String, obj: [String: Any]) {
        processMovementPrediction(
            pred: pred,
            obj: obj,
            tracker: &lungesErrors,
            goodPrefix: "correct",
            targetReps: lungesTargetCorrectReps,
            onComplete: { [weak self] in
                guard let self else { return }
                self.speech.speak("Lunges complete.", language: self.speechLang, minInterval: 0)
                self.finishSession()
            }
        )
    }

    // MARK: - Shared Movement Rep Handler

    /// Shared handler for movement exercises (squat, lunges) that counts reps and tracks errors.
    ///
    /// - Parameters:
    ///   - pred: Lowercase prediction label from the backend.
    ///   - obj: Full JSON message dictionary.
    ///   - tracker: The exercise-specific error tracker to update.
    ///   - goodPrefix: The label prefix indicating a correct rep (e.g., "good" for squat, "correct" for lunges).
    ///   - targetReps: Number of correct reps needed to complete the exercise.
    ///   - onComplete: Closure called when the target is reached.
    private func processMovementPrediction(
        pred: String,
        obj: [String: Any],
        tracker: inout ExerciseErrorTracker,
        goodPrefix: String,
        targetReps: Int,
        onComplete: @escaping () -> Void
    ) {
        let reps = obj["reps"] as? [String: Any]
        let totalFromPayload = reps?["total"] as? Int
        if let totalFromPayload {
            self.totalReps = totalFromPayload
        }

        let currentRepNumber: Int? = {
            let rep = totalFromPayload ?? self.totalReps
            return rep > 0 ? rep : nil
        }()

        let isGoodRep = pred.hasPrefix(goodPrefix)
        if !isGoodRep {
            if shouldTrackAsMistakeLabel(pred) {
                tracker.recordMistakeIfTransitioned(
                    prediction: pred,
                    elapsedSeconds: sessionElapsedSecond(),
                    repNumber: currentRepNumber,
                    displayLabel: Self.displayLabel(for: pred)
                )
            }
        } else {
            tracker.markPredictionAsCorrect()
        }

        // Update rep counts from the payload
        if let reps {
            let correct: Int? = (reps["correct"] as? Int) ?? (reps["good"] as? Int)
            if let correct = correct {
                if correct > self.correctReps {
                    self.speech.speak("Good", language: self.speechLang, minInterval: 0.5, allowRepeat: true)
                    self.lastSpokenCorrectReps = correct
                }
                self.correctReps = correct
            }

            let incorrect: Int? = (reps["incorrect"] as? Int) ?? (reps["bad"] as? Int)
            if let incorrect = incorrect {
                self.incorrectReps = incorrect
            }

            if self.correctReps >= targetReps {
                onComplete()
            }
        }
    }

    // MARK: - Session Summary Builder

    /// Builds the complete session summary from accumulated exercise data.
    private func buildSessionSummary() -> SessionSummary {
        let totalTime = Int(Date().timeIntervalSince(startTime))
        var calories = max(0, correctReps) * 4
        if initialMode == .plank {
            calories = Int(plankHold.elapsedSeconds * 0.5)
        }

        var items: [ExerciseSummaryItem] = []

        if initialMode == .plank {
            items.append(.isometric(
                name: "Plank",
                durationSeconds: plankHold.elapsedSeconds,
                targetSeconds: plankHold.targetSeconds,
                errors: buildErrorCounts(from: plankErrors.errorCounts),
                mistakes: plankErrors.mistakeEvents
            ))
        } else if initialMode == .lunges {
            items.append(.movement(
                name: "Lunges",
                totalReps: totalReps,
                correctReps: correctReps,
                incorrectReps: incorrectReps,
                targetCorrectReps: lungesTargetCorrectReps,
                errors: buildErrorCounts(from: lungesErrors.errorCounts),
                mistakes: lungesErrors.mistakeEvents
            ))
        } else {
            // Multi-exercise program: wall-sit -> squat -> plank
            items.append(.isometric(
                name: "Wall-Sit",
                durationSeconds: wallSitHold.elapsedSeconds,
                targetSeconds: wallSitHold.targetSeconds,
                errors: buildErrorCounts(from: wallSitErrors.errorCounts),
                mistakes: wallSitErrors.mistakeEvents
            ))

            items.append(.movement(
                name: "Squat",
                totalReps: totalReps,
                correctReps: correctReps,
                incorrectReps: incorrectReps,
                targetCorrectReps: squatTargetCorrectReps,
                errors: buildErrorCounts(from: squatErrors.errorCounts),
                mistakes: squatErrors.mistakeEvents
            ))

            items.append(.isometric(
                name: "Plank",
                durationSeconds: plankHold.elapsedSeconds,
                targetSeconds: plankHold.targetSeconds,
                errors: buildErrorCounts(from: plankErrors.errorCounts),
                mistakes: plankErrors.mistakeEvents
            ))
        }

        return SessionSummary(items: items, totalTimeSeconds: totalTime, estimatedCalories: calories)
    }

    /// Converts an error count dictionary into sorted `ErrorCount` display models.
    private func buildErrorCounts(from counts: [String: Int]) -> [ErrorCount] {
        counts
            .filter { $0.value > 0 }
            .map { ErrorCount(reason: Self.displayLabel(for: $0.key), count: $0.value) }
            .sorted { $0.count > $1.count }
    }

    // MARK: - Label Formatting

    /// Converts a backend prediction label to human-readable display text.
    /// Handles known labels explicitly and falls back to title-casing unknown labels.
    private static func displayLabel(for backendLabel: String) -> String {
        let p = backendLabel.lowercased()
        if p.contains("feet_too_close") || p.contains("feet too close") { return "Feet too close" }
        if p.contains("knee_ins") || p.contains("knees_in") || p.contains("knees in") { return "Knees in" }
        if p.contains("round_back") { return "Round back" }
        if p.contains("torso_lean_forward") || p.contains("torso lean forward") { return "Torso lean forward" }
        if p.contains("knee_over_toe") || p.contains("knee over toe") { return "Knee over toe" }
        if p.contains("not_deep_enough") { return "Not deep enough" }
        if p.contains("stand_too_narrow") { return "Stand too narrow" }
        if p.contains("stand_too_wide") { return "Stand too wide" }
        // Fallback: replace underscores and title-case each word
        let withSpaces = backendLabel.replacingOccurrences(of: "_", with: " ")
        guard !withSpaces.isEmpty else { return backendLabel }
        return withSpaces.split(separator: " ").map { $0.prefix(1).uppercased() + $0.dropFirst().lowercased() }.joined(separator: " ")
    }

    // MARK: - Helpers

    /// Returns the number of seconds elapsed since the session started.
    private func sessionElapsedSecond() -> Int {
        max(0, Int(Date().timeIntervalSince(startTime)))
    }

    /// Determines if a prediction label represents a trackable mistake (not a status/correct label).
    private func shouldTrackAsMistakeLabel(_ pred: String) -> Bool {
        let p = pred.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if p.isEmpty || p == "..." { return false }
        if p.contains("correct") { return false }
        if p == "good" || p == "passed" || p == "stand" || p == "good_stand" || p == "good_squat" { return false }
        return true
    }

    /// Generates appropriate feedback text for an incorrect prediction.
    private func incorrectFeedbackText(for pred: String) -> String {
        let trimmed = pred.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, trimmed != "..." else {
            return "Adjust your position"
        }
        return Self.displayLabel(for: trimmed)
    }

    /// Updates feedback only when the value actually changes (prevents unnecessary UI updates).
    private func setFeedbackIfChanged(_ newText: String) {
        if feedback != newText {
            feedback = newText
        }
    }
}
