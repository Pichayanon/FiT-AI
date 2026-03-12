import Foundation
import SwiftUI

/// Coordinates workout session state, backend messaging, and summary generation.
@MainActor
final class WorkoutSessionViewModel: ObservableObject {
    // ✅ Backend phases (match Python)
    private enum BackendPhase: String {
        case NO_POSE, HAVE_POSE, BUFFERING, INFERENCING
    }

    // MARK: - Dependencies
    @Published var cameraManager = CameraManager()
    let speech = SpeechManager()
    private let workoutHistory = WorkoutHistoryService()

    // MARK: - Inputs
    let setTitle: String
    private let program: WorkoutProgramDefinition

    // MARK: - Published States (ย้ายมาจาก View เดิม)
    @Published var feedback: String = "Side view • Stand → Lower → Hold"
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

    // Wall-sit hold progress
    @Published var correctSeconds: Double = 0
    @Published var targetSeconds: Double = 5.0
    @Published var passedWallSit: Bool = false

    // Gate: ต้อง correct 3 ครั้งติด
    @Published var wallSitConsecutiveCorrect: Int = 0
    @Published var wallSitCountingActive: Bool = false
    @Published var wallSitIsCorrectHold: Bool = false

    // Plank hold progress
    @Published var plankCorrectSeconds: Double = 0
    @Published var plankTargetSeconds: Double = 5.0
    @Published var passedPlank: Bool = false

    // Gate: ต้อง correct 3 ครั้งติด (สำหรับ Plank)
    @Published var plankConsecutiveCorrect: Int = 0
    @Published var plankCountingActive: Bool = false
    @Published var plankIsCorrectHold: Bool = false

    // Squat preview
    @Published var showSquatPreview: Bool = false
    @Published var squatPreviewSeconds: Int = 8

    // Plank preview (before performing plank, after squat)
    @Published var showPlankPreview: Bool = false
    @Published var plankPreviewSeconds: Int = 8

    // Auto switch control
    @Published var didSwitchToSquat: Bool = false
    @Published var didSwitchToPlank: Bool = false

    // Squat target: correct 3 reps = finish
    @Published var squatTargetCorrectReps: Int = 3

    // Lunges target
    @Published var lungesTargetCorrectReps: Int = 3

    // Current mode (sequence: wall-sit → squat → plank)
    @Published var mode: WorkoutSessionMode = .wallSit

    /// สรุป session สำหรับส่งไปหน้า result (เซ็ตตอน finishSession)
    @Published var sessionSummary: SessionSummary = SessionSummary(items: [], totalTimeSeconds: 0, estimatedCalories: 0)

    /// นับจำนวนครั้งที่ผิดแต่ละประเภทใน wall-sit (label จาก backend → count)
    private var wallSitErrorCounts: [String: Int] = [:]
    /// ไทม์ไลน์ว่าผิดตอนไหน + ผิดอะไร
    private var wallSitMistakeEvents: [MistakeEvent] = []
    /// ใช้ dedupe: นับ error แค่ตอนเปลี่ยนจาก correct → error (ไม่นับทุก message)
    private var lastWallSitPredWasCorrect: Bool = true

    /// นับจำนวนครั้งที่ผิดแต่ละประเภทใน squat (label จาก backend → count)
    private var squatErrorCounts: [String: Int] = [:]
    /// ไทม์ไลน์ว่าผิดตอนไหน + ผิดอะไร
    private var squatMistakeEvents: [MistakeEvent] = []
    /// ใช้ dedupe: นับ error แค่ตอนเปลี่ยนจาก correct → error
    private var lastSquatPredWasCorrect: Bool = true

    /// นับจำนวนครั้งที่ผิดแต่ละประเภทใน plank (label จาก backend → count)
    private var plankErrorCounts: [String: Int] = [:]
    /// ไทม์ไลน์ว่าผิดตอนไหน + ผิดอะไร
    private var plankMistakeEvents: [MistakeEvent] = []
    /// ใช้ dedupe: นับ error แค่ตอนเปลี่ยนจาก correct → error (ไม่นับทุก message)
    private var lastPlankPredWasCorrect: Bool = true

    // Lunges stats
    private var lungesErrorCounts: [String: Int] = [:]
    private var lungesMistakeEvents: [MistakeEvent] = []
    private var lastLungesPredWasCorrect: Bool = true

    // MARK: - Constants
    private let speechLang = "en-US"
    private let previewDurationSeconds = 8

    // WS endpoints
    private let wsWallSitURL = "ws://172.20.10.5:5050/ws/video"
    private let wsSquatURL   = "ws://172.20.10.5:5051/ws/video"
    private let wsPlankURL   = "ws://172.20.10.5:5052/ws/video"
    private let wsLungesURL  = "ws://172.20.10.5:5053/ws/video"

    private var initialMode: WorkoutSessionMode {
        program.initialMode
    }

    private var activeWSURL: String {
        switch mode {
        case .wallSit: return wsWallSitURL
        case .squat:   return wsSquatURL
        case .plank:   return wsPlankURL
        case .lunges:  return wsLungesURL
        }
    }

    // ปรับ threshold ตามโมเดลจริงของคุณ
    private let wallSitConfThreshold: Double = 0.50
    private let plankConfThreshold: Double = 0.50

    // เปิด/ปิดปุ่มทดสอบเสียง (ถ้าจะใช้ใน View ให้ expose เพิ่ม)
    let showTestVoiceButton: Bool = false

    // MARK: - Derived
    var wallSitProgress01: Double {
        guard targetSeconds > 0 else { return 0 }
        return min(1.0, max(0.0, correctSeconds / targetSeconds))
    }

    var squatProgress01: Double {
        let tgt = Double(max(1, squatTargetCorrectReps))
        return min(1.0, max(0.0, Double(correctReps) / tgt))
    }

    var plankProgress01: Double {
        guard plankTargetSeconds > 0 else { return 0 }
        return min(1.0, max(0.0, plankCorrectSeconds / plankTargetSeconds))
    }

    var lungesProgress01: Double {
        let tgt = Double(max(1, lungesTargetCorrectReps))
        return min(1.0, max(0.0, Double(correctReps) / tgt))
    }

    var titleForHUD: String {
        "\(setTitle) • \(mode.displayName)"
    }

    init(setTitle: String) {
        let program = WorkoutCatalog.program(for: setTitle)
        self.program = program
        self.setTitle = program.title
        self.mode = program.initialMode
    }

    // MARK: - Lifecycle
    func onAppear() {
        speech.configureAudioSession()
        cameraManager.startSession()
        cameraManager.onBackendMessage = { [weak self] text in
            self?.handleBackendMessage(text)
        }
    }

    func onDisappear() {
        stopSession()
        cameraManager.stopSession()
    }

    // MARK: - Session Control
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

    func stopSession() {
        isSessionRunning = false
        cameraManager.stopStreaming()
        backendState = BackendPhase.NO_POSE.rawValue
        speech.stop()
    }

    /// Reset all session-scoped counters before connecting to a workout backend.
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

        correctSeconds = 0
        targetSeconds = 5.0
        wallSitConsecutiveCorrect = 0
        wallSitCountingActive = false
        wallSitIsCorrectHold = false

        plankCorrectSeconds = 0
        plankTargetSeconds = 5.0
        plankConsecutiveCorrect = 0
        plankCountingActive = false
        plankIsCorrectHold = false
        passedPlank = false

        squatStandOK = false
        squatStarted = false
        lastStandOKAt = nil
        lastPleaseStartAt = .distantPast

        lungesErrorCounts = [:]
        lungesMistakeEvents = []
        lastLungesPredWasCorrect = true

        wallSitErrorCounts = [:]
        wallSitMistakeEvents = []
        squatErrorCounts = [:]
        squatMistakeEvents = []
        plankErrorCounts = [:]
        plankMistakeEvents = []
        lastWallSitPredWasCorrect = true
        lastSquatPredWasCorrect = true
        lastPlankPredWasCorrect = true
    }

    /// Speak the exercise setup instructions once after the stream begins.
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

    func finishSession() {
        stopSession()
        speech.speak("Session complete", language: speechLang, minInterval: 0)
        sessionSummary = buildSessionSummary()
        Task { await workoutHistory.saveSession(summary: sessionSummary, setTitle: setTitle) }
        navigateToResult = true
    }

    /// Map backend label to display text (English); removes underscores and title-cases.
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
        // Replace underscores with spaces and capitalize each word (e.g. hips_too_high → Hips Too High)
        let withSpaces = backendLabel.replacingOccurrences(of: "_", with: " ")
        guard !withSpaces.isEmpty else { return backendLabel }
        return withSpaces.split(separator: " ").map { $0.prefix(1).uppercased() + $0.dropFirst().lowercased() }.joined(separator: " ")
    }

    private func buildSessionSummary() -> SessionSummary {
        let totalTime = Int(Date().timeIntervalSince(startTime))
        var calories = max(0, correctReps) * 4
        if initialMode == .plank {
            calories = Int(plankCorrectSeconds * 0.5)
        }

        var items: [ExerciseSummaryItem] = []

        if initialMode == .plank {
            items.append(.isometric(
                name: "Plank",
                durationSeconds: plankCorrectSeconds,
                targetSeconds: plankTargetSeconds,
                errors: buildErrorCounts(from: plankErrorCounts),
                mistakes: plankMistakeEvents
            ))
        } else if initialMode == .lunges {
            items.append(.movement(
                name: "Lunges",
                totalReps: totalReps,
                correctReps: correctReps,
                incorrectReps: incorrectReps,
                targetCorrectReps: lungesTargetCorrectReps,
                errors: buildErrorCounts(from: lungesErrorCounts),
                mistakes: lungesMistakeEvents
            ))
        } else {
            items.append(.isometric(
                name: "Wall-Sit",
                durationSeconds: correctSeconds,
                targetSeconds: targetSeconds,
                errors: buildErrorCounts(from: wallSitErrorCounts),
                mistakes: wallSitMistakeEvents
            ))

            items.append(.movement(
                name: "Squat",
                totalReps: totalReps,
                correctReps: correctReps,
                incorrectReps: incorrectReps,
                targetCorrectReps: squatTargetCorrectReps,
                errors: buildErrorCounts(from: squatErrorCounts),
                mistakes: squatMistakeEvents
            ))

            items.append(.isometric(
                name: "Plank",
                durationSeconds: plankCorrectSeconds,
                targetSeconds: plankTargetSeconds,
                errors: buildErrorCounts(from: plankErrorCounts),
                mistakes: plankMistakeEvents
            ))
        }

        return SessionSummary(items: items, totalTimeSeconds: totalTime, estimatedCalories: calories)
    }

    private func buildErrorCounts(from counts: [String: Int]) -> [ErrorCount] {
        counts
            .filter { $0.value > 0 }
            .map { ErrorCount(reason: Self.displayLabel(for: $0.key), count: $0.value) }
            .sorted { $0.count > $1.count }
    }

    // MARK: - Timers
    func handleWallSitTick() {
        guard isSessionRunning else { return }
        guard mode == .wallSit else { return }
        guard wallSitCountingActive else { return }
        guard !passedWallSit else { return }
        guard wallSitIsCorrectHold else { return }
        guard !showSquatPreview else { return } // ขึ้น preview แล้วไม่ต้องนับต่อ

        correctSeconds = min(targetSeconds, correctSeconds + 0.1)

        if correctSeconds >= targetSeconds {
            correctSeconds = targetSeconds
            passedWallSit = true
            setFeedbackIfChanged("Passed ✅")
            speech.speak("Wall-sit complete. Switch to Squat.", language: speechLang, minInterval: 0)

            // ✅ หยุด stream ระหว่าง preview กันมัน detect ต่อ
            cameraManager.stopStreaming()

            startSquatPreviewThenSwitch()
        }
    }

    func handleSquatIdleTick() {
        guard isSessionRunning else { return }
        guard mode == .squat else { return }
        guard squatStandOK else { return }
        guard !squatStarted else { return }

        guard let okAt = lastStandOKAt else { return }

        // รอ 30 วิ หลัง Stand OK
        if Date().timeIntervalSince(okAt) > 30.0 {
            // กันพูดรัว (พูดทุก ~15 วิ)
            if Date().timeIntervalSince(lastPleaseStartAt) > 15.0 {
                speech.speak("Please start squat", language: speechLang, minInterval: 0)
                lastPleaseStartAt = Date()
            }
        }
    }

    // MARK: - Feedback speech
    func handleFeedbackChange() {
        guard isSessionRunning else { return }
        let cleaned = speech.cleanForSpeech(feedback)
        guard speech.shouldSpeak(cleaned) else { return }
        speech.speak(cleaned, language: speechLang, minInterval: 1.2)
    }

    // MARK: - Squat preview / switch
    private func startSquatPreviewThenSwitch() {
        showSquatPreview = true
        squatPreviewSeconds = previewDurationSeconds

        speech.speak(
            WorkoutSessionMode.squat.demoInstructionText,
            language: speechLang,
            minInterval: 0,
            allowRepeat: true
        )

        // countdown UI
        for i in 1...previewDurationSeconds {
            DispatchQueue.main.asyncAfter(deadline: .now() + Double(i)) { [weak self] in
                guard let self else { return }
                if self.showSquatPreview {
                    self.squatPreviewSeconds = max(0, self.previewDurationSeconds - i)
                }
            }
        }

        DispatchQueue.main.asyncAfter(deadline: .now() + Double(previewDurationSeconds)) { [weak self] in
            guard let self else { return }
            self.showSquatPreview = false
            self.switchToSquat()
        }
    }

    // MARK: - Plank preview / switch (after squat)
    private func startPlankPreviewThenSwitch() {
        showPlankPreview = true
        plankPreviewSeconds = previewDurationSeconds

        speech.speak(
            WorkoutSessionMode.plank.demoInstructionText,
            language: speechLang,
            minInterval: 0,
            allowRepeat: true
        )

        for i in 1...previewDurationSeconds {
            DispatchQueue.main.asyncAfter(deadline: .now() + Double(i)) { [weak self] in
                guard let self else { return }
                if self.showPlankPreview {
                    self.plankPreviewSeconds = max(0, self.previewDurationSeconds - i)
                }
            }
        }

        DispatchQueue.main.asyncAfter(deadline: .now() + Double(previewDurationSeconds)) { [weak self] in
            guard let self else { return }
            self.showPlankPreview = false
            self.switchToPlank()
        }
    }

    private func switchToSquat() {
        guard isSessionRunning else { return }
        guard !didSwitchToSquat else { return }

        didSwitchToSquat = true
        mode = .squat
        backendState = BackendPhase.NO_POSE.rawValue
        setFeedbackIfChanged("PASSED (Switch to Squat)")

        // reset squat counters (เริ่มนับใหม่)
        totalReps = 0
        correctReps = 0
        incorrectReps = 0
        lastSpokenCorrectReps = 0

        // squat helpers
        squatStandOK = false
        squatStarted = false
        lastStandOKAt = nil
        lastPleaseStartAt = .distantPast

        // กัน wall-sit timer สะสมต่อหลังสลับไป squat
        wallSitCountingActive = false
        wallSitIsCorrectHold = false

        // start squat streaming
        cameraManager.stopStreaming()
        cameraManager.startStreaming(to: activeWSURL)

    }

    private func switchToPlank() {
        guard isSessionRunning else { return }
        guard !didSwitchToPlank else { return }

        didSwitchToPlank = true
        mode = .plank
        backendState = BackendPhase.NO_POSE.rawValue
        setFeedbackIfChanged("PASSED (Switch to Plank)")

        // reset plank counters
        plankCorrectSeconds = 0
        plankTargetSeconds = 5.0
        plankConsecutiveCorrect = 0
        plankCountingActive = false
        plankIsCorrectHold = false
        passedPlank = false
        lastPlankPredWasCorrect = true

        cameraManager.stopStreaming()
        cameraManager.startStreaming(to: activeWSURL)

    }

    // MARK: - Backend Message
    private func handleBackendMessage(_ text: String) {
        guard let data = text.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return
        }

        let type = obj["type"] as? String ?? ""

        if type == "status" {
            let stateRaw = (obj["state"] as? String ?? BackendPhase.NO_POSE.rawValue).uppercased()
            let msg = obj["message"] as? String

            DispatchQueue.main.async {
                self.backendState = stateRaw

                // ถ้า NO_POSE ให้ reset hold/ready ต่าง ๆ (กัน timer นับค้าง)
                if stateRaw == BackendPhase.NO_POSE.rawValue {
                    if self.mode == .wallSit {
                        self.wallSitIsCorrectHold = false
                        self.wallSitCountingActive = false
                        self.wallSitConsecutiveCorrect = 0
                        self.correctSeconds = 0
                    } else if self.mode == .plank {
                        self.plankIsCorrectHold = false
                        self.plankCountingActive = false
                        self.plankConsecutiveCorrect = 0
                        self.plankCorrectSeconds = 0
                    }
                }

                if let msg { self.setFeedbackIfChanged(msg) }
            }
            return
        }

        if type == "info" {
            let msg = obj["message"] as? String ?? "..."
            DispatchQueue.main.async { self.setFeedbackIfChanged(msg) }
            return
        }

        if type == "result" {
            let predRaw = (obj["prediction"] as? String ?? "...")
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased()

            let conf = obj["confidence"] as? Double

            DispatchQueue.main.async {
                // แสดง feedback (อย่าให้รัวเกินไปจน UI ดูสั่น)
                if let conf {
                    self.setFeedbackIfChanged("\(predRaw) • \(String(format: "%.2f", conf))")
                } else {
                    self.setFeedbackIfChanged("\(predRaw)")
                }

                // WALL-SIT
                if self.mode == .wallSit {
                    self.handleWallSitResult(pred: predRaw, conf: conf)
                    return
                }

                // SQUAT
                if self.mode == .squat {
                    let backendMode = obj["mode"] as? String ?? ""

                    // 1️⃣ STAND result (mode == "stand")
                    if backendMode == "stand" {
                        let standOk = obj["stand_ok"] as? Bool ?? false
                        if standOk {
                            self.lastSquatPredWasCorrect = true
                            if !self.squatStandOK {
                                self.squatStandOK = true
                                self.lastStandOKAt = Date()
                                self.speech.speak("Stand OK", language: self.speechLang, minInterval: 0)
                            }
                        } else {
                            // stand_too_narrow / stand_too_wide
                            self.speech.speak(Self.displayLabel(for: predRaw), language: self.speechLang, minInterval: 2.0)
                        }
                        return
                    }

                    // 2️⃣ BOTTOM result (mode == "bottom")
                    self.squatStarted = true

                    let reps = obj["reps"] as? [String: Any]
                    let totalFromPayload = reps?["total"] as? Int
                    if let totalFromPayload {
                        self.totalReps = totalFromPayload
                    }

                    let currentRepNumber: Int? = {
                        let rep = totalFromPayload ?? self.totalReps
                        return rep > 0 ? rep : nil
                    }()

                    let isGoodRep = predRaw.hasPrefix("good")
                    if !isGoodRep {
                        // knee_ins / round_back / not_deep_enough
                        if self.lastSquatPredWasCorrect {
                            self.squatErrorCounts[predRaw] = (self.squatErrorCounts[predRaw] ?? 0) + 1
                            self.squatMistakeEvents.append(
                                MistakeEvent(
                                    atSecond: self.sessionElapsedSecond(),
                                    reason: Self.displayLabel(for: predRaw),
                                    repNumber: currentRepNumber
                                )
                            )
                        }
                        self.lastSquatPredWasCorrect = false
                    } else {
                        self.lastSquatPredWasCorrect = true
                    }

                    // 3️⃣ REP UPDATE
                    if let reps {
                        let correct: Int? = (reps["correct"] as? Int)
                            ?? (reps["good"] as? Int)
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

                        if self.correctReps >= self.squatTargetCorrectReps {
                            self.speech.speak("Squat complete. Switch to Plank.", language: self.speechLang, minInterval: 0)
                            self.cameraManager.stopStreaming()
                            self.startPlankPreviewThenSwitch()
                        }
                    }
                    return
                }

                // PLANK
                if self.mode == .plank {
                    self.handlePlankResult(pred: predRaw, conf: conf)
                }

                // LUNGES
                if self.mode == .lunges {
                    // Logic similar to Squat (counting reps)
                    let reps = obj["reps"] as? [String: Any]
                    let totalFromPayload = reps?["total"] as? Int
                    if let totalFromPayload {
                        self.totalReps = totalFromPayload
                    }

                    let currentRepNumber: Int? = {
                        let rep = totalFromPayload ?? self.totalReps
                        return rep > 0 ? rep : nil
                    }()

                    let isGoodRep = predRaw.hasPrefix("correct") // "correct" is the label for good lunges
                    if !isGoodRep {
                        // feet_too_forward, knee_over_toe, not_deep_enough, rear_knee_hit_floor
                        if self.lastLungesPredWasCorrect {
                            self.lungesErrorCounts[predRaw] = (self.lungesErrorCounts[predRaw] ?? 0) + 1
                            self.lungesMistakeEvents.append(
                                MistakeEvent(
                                    atSecond: self.sessionElapsedSecond(),
                                    reason: Self.displayLabel(for: predRaw),
                                    repNumber: currentRepNumber
                                )
                            )
                        }
                        self.lastLungesPredWasCorrect = false
                    } else {
                        self.lastLungesPredWasCorrect = true
                    }

                    if let reps {
                        let correct: Int? = (reps["correct"] as? Int)
                        if let correct = correct {
                            if correct > self.correctReps {
                                self.speech.speak("Good", language: self.speechLang, minInterval: 0.5, allowRepeat: true)
                                self.lastSpokenCorrectReps = correct
                            }
                            self.correctReps = correct
                        }

                        let incorrect: Int? = (reps["incorrect"] as? Int)
                        if let incorrect = incorrect {
                            self.incorrectReps = incorrect
                        }

                        if self.correctReps >= self.lungesTargetCorrectReps {
                            self.speech.speak("Lunges complete.", language: self.speechLang, minInterval: 0)
                            self.finishSession()
                        }
                    }
                }
            }
            return
        }
    }

    // MARK: - Wall-sit logic (3x correct gate -> then 5s hold timer)
    private func handleWallSitResult(pred: String, conf: Double?) {
        // รองรับ label แบบ dataset_correct / correct / passed ฯลฯ
        let p = pred.lowercased()
        let c = conf ?? 0.0

        // ถือว่า "correct" ถ้า label มีคำว่า correct และ confidence ผ่าน threshold
        let isCorrectNow = p.contains("correct") && c >= wallSitConfThreshold

        // ให้ timer รู้ว่าตอนนี้ correct อยู่ไหม (แม้ backend dedup)
        wallSitIsCorrectHold = isCorrectNow

        // นับข้อผิดพลาดแต่ละประเภท (สำหรับ summary) — เฉพาะตอนเปลี่ยนจาก correct → error (ไม่นับทุก message)
        if isCorrectNow {
            lastWallSitPredWasCorrect = true
        } else {
            if lastWallSitPredWasCorrect && shouldTrackAsMistakeLabel(pred) {
                wallSitErrorCounts[pred] = (wallSitErrorCounts[pred] ?? 0) + 1
                wallSitMistakeEvents.append(
                    MistakeEvent(atSecond: sessionElapsedSecond(), reason: Self.displayLabel(for: pred), repNumber: nil)
                )
            }
            lastWallSitPredWasCorrect = false
        }

        if passedWallSit { return }

        // ---- Phase 1: ต้องได้ correct 3 ครั้งติดก่อน ----
        if !wallSitCountingActive {
            if isCorrectNow {
                wallSitConsecutiveCorrect = min(3, wallSitConsecutiveCorrect + 1)

                // พูดแค่ครั้งแรกที่เริ่มเข้า correct streak
                if wallSitConsecutiveCorrect == 1 {
                    speech.speak("OK", language: speechLang, minInterval: 0)
                }

                // ถึง 3 ครั้งติด => เริ่มนับเวลา
                if wallSitConsecutiveCorrect >= 3 {
                    wallSitCountingActive = true
                    correctSeconds = 0
                    targetSeconds = 5.0
                    setFeedbackIfChanged("Start hold…")
                } else {
                    setFeedbackIfChanged("Correct \(wallSitConsecutiveCorrect)/3 • Keep holding")
                }
            } else {
                // หลุด correct ก่อนครบ 3 => reset streak
                if wallSitConsecutiveCorrect != 0 {
                    wallSitConsecutiveCorrect = 0
                    setFeedbackIfChanged(incorrectFeedbackText(for: pred))
                }
            }
            return
        }

        // ---- Phase 2: กำลังนับเวลา 5s ----
        if !isCorrectNow {
            // หลุดตอนกำลังนับ => reset ทั้งหมด
            wallSitCountingActive = false
            wallSitIsCorrectHold = false
            wallSitConsecutiveCorrect = 0
            correctSeconds = 0
            setFeedbackIfChanged(incorrectFeedbackText(for: pred))
        }
    }

    // MARK: - Plank logic (เหมือน wall-sit: 3x correct gate -> 5s hold)
    func handlePlankTick() {
        guard isSessionRunning else { return }
        guard mode == .plank else { return }
        guard plankCountingActive else { return }
        guard !passedPlank else { return }
        guard plankIsCorrectHold else { return }

        plankCorrectSeconds = min(plankTargetSeconds, plankCorrectSeconds + 0.1)

        if plankCorrectSeconds >= plankTargetSeconds {
            plankCorrectSeconds = plankTargetSeconds
            passedPlank = true
            setFeedbackIfChanged("Plank completed ✅")
            finishSession()
        }
    }

    private func handlePlankResult(pred: String, conf: Double?) {
        let p = pred.lowercased()
        let c = conf ?? 0.0

        let isCorrectNow = p.contains("correct") && c >= plankConfThreshold

        plankIsCorrectHold = isCorrectNow

        // นับข้อผิดพลาดแต่ละประเภท — เฉพาะตอนเปลี่ยนจาก correct → error (ไม่นับทุก message)
        if isCorrectNow {
            lastPlankPredWasCorrect = true
        } else {
            if lastPlankPredWasCorrect && shouldTrackAsMistakeLabel(pred) {
                plankErrorCounts[pred] = (plankErrorCounts[pred] ?? 0) + 1
                plankMistakeEvents.append(
                    MistakeEvent(atSecond: sessionElapsedSecond(), reason: Self.displayLabel(for: pred), repNumber: nil)
                )
            }
            lastPlankPredWasCorrect = false
        }

        if passedPlank { return }

        if !plankCountingActive {
            if isCorrectNow {
                plankConsecutiveCorrect = min(3, plankConsecutiveCorrect + 1)

                if plankConsecutiveCorrect == 1 {
                    speech.speak("OK", language: speechLang, minInterval: 0)
                }

                if plankConsecutiveCorrect >= 3 {
                    plankCountingActive = true
                    plankCorrectSeconds = 0
                    plankTargetSeconds = 5.0
                    setFeedbackIfChanged("Start plank hold…")
                } else {
                    setFeedbackIfChanged("Plank correct \(plankConsecutiveCorrect)/3 • Keep holding")
                }
            } else {
                if plankConsecutiveCorrect != 0 {
                    plankConsecutiveCorrect = 0
                    setFeedbackIfChanged(incorrectFeedbackText(for: pred))
                }
            }
            return
        }

        if !isCorrectNow {
            plankCountingActive = false
            plankIsCorrectHold = false
            plankConsecutiveCorrect = 0
            plankCorrectSeconds = 0
            setFeedbackIfChanged(incorrectFeedbackText(for: pred))
        }
    }

    // MARK: - Helpers
    private func sessionElapsedSecond() -> Int {
        max(0, Int(Date().timeIntervalSince(startTime)))
    }

    private func shouldTrackAsMistakeLabel(_ pred: String) -> Bool {
        let p = pred.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if p.isEmpty || p == "..." { return false }
        if p.contains("correct") { return false }
        if p == "good" || p == "passed" || p == "stand" || p == "good_stand" || p == "good_squat" { return false }
        return true
    }

    private func incorrectFeedbackText(for pred: String) -> String {
        let trimmed = pred.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, trimmed != "..." else {
            return "Adjust your position"
        }
        return Self.displayLabel(for: trimmed)
    }

    private func setFeedbackIfChanged(_ newText: String) {
        if feedback != newText {
            feedback = newText
        }
    }
}
