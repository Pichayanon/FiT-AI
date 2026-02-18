import Foundation
import SwiftUI

/// ViewModel สำหรับจัดการ state + business logic ของ `WorkoutSessionView`
@MainActor
final class WorkoutSessionViewModel: ObservableObject {
    enum Mode { case wallSit, squat, plank }

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
    @Published var squatPreviewSeconds: Int = 5

    // Auto switch control
    @Published var didSwitchToSquat: Bool = false
    @Published var didSwitchToPlank: Bool = false

    // Squat target: correct 3 reps = finish
    @Published var squatTargetCorrectReps: Int = 3

    // Current mode (เริ่มจาก wall-sit)
    @Published var mode: Mode = .wallSit

    /// สรุป session สำหรับส่งไปหน้า result (เซ็ตตอน finishSession)
    @Published var sessionSummary: SessionSummary = SessionSummary(items: [], totalTimeSeconds: 0, estimatedCalories: 0)

    /// นับจำนวนครั้งที่ผิดแต่ละประเภทใน wall-sit (label จาก backend → count)
    private var wallSitErrorCounts: [String: Int] = [:]

    /// นับจำนวนครั้งที่ผิดแต่ละประเภทใน plank (label จาก backend → count)
    private var plankErrorCounts: [String: Int] = [:]

    // MARK: - Constants
    private let speechLang = "en-US"

    // WS endpoints
    private let wsWallSitURL = "ws://172.20.10.5:5050/ws/video"
    private let wsSquatURL   = "ws://172.20.10.5:5051/ws/video"
    private let wsPlankURL   = "ws://172.20.10.5:5052/ws/video"

    private var activeWSURL: String {
        switch mode {
        case .wallSit: return wsWallSitURL
        case .squat:   return wsSquatURL
        case .plank:   return wsPlankURL
        }
    }

    // ปรับ threshold ตามโมเดลจริงของคุณ
    private let wallSitConfThreshold: Double = 0.50

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

    var titleForHUD: String {
        switch mode {
        case .wallSit: return "\(setTitle) • Wall-Sit"
        case .squat:   return "\(setTitle) • Squat"
        case .plank:   return "\(setTitle) • Plank"
        }
    }

    init(setTitle: String) {
        self.setTitle = setTitle
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

        // reset ALL
        mode = .wallSit
        didSwitchToSquat = false
        didSwitchToPlank = false
        passedWallSit = false

        // preview
        showSquatPreview = false
        squatPreviewSeconds = 5

        // reps (จริง ๆ ใช้ตอน squat)
        totalReps = 0
        correctReps = 0
        incorrectReps = 0
        lastSpokenCorrectReps = 0

        // wall-sit progress
        correctSeconds = 0
        targetSeconds = 5.0
        wallSitConsecutiveCorrect = 0
        wallSitCountingActive = false
        wallSitIsCorrectHold = false

        // plank progress
        plankCorrectSeconds = 0
        plankTargetSeconds = 5.0
        plankConsecutiveCorrect = 0
        plankCountingActive = false
        plankIsCorrectHold = false
        passedPlank = false

        // squat helpers
        squatStandOK = false
        squatStarted = false
        lastStandOKAt = nil
        lastPleaseStartAt = .distantPast

        wallSitErrorCounts = [:]
        plankErrorCounts = [:]

        cameraManager.startStreaming(to: activeWSURL)
        speech.speak("Session started", language: speechLang, minInterval: 0)
    }

    func stopSession() {
        isSessionRunning = false
        cameraManager.stopStreaming()
        backendState = BackendPhase.NO_POSE.rawValue
        speech.stop()
    }

    func finishSession() {
        stopSession()
        sessionSummary = buildSessionSummary()
        Task { await workoutHistory.saveSession(summary: sessionSummary, setTitle: setTitle) }
        navigateToResult = true
    }

    /// Map backend label to display text (English)
    private static func displayLabel(for backendLabel: String) -> String {
        let p = backendLabel.lowercased()
        if p.contains("feet_too_close") || p.contains("feet too close") { return "Feet too close" }
        if p.contains("knees_in") || p.contains("knees in") { return "Knees in" }
        return backendLabel
    }

    private func buildSessionSummary() -> SessionSummary {
        let totalTime = Int(Date().timeIntervalSince(startTime))
        let calories = max(0, correctReps) * 4

        var items: [ExerciseSummaryItem] = []

        // Wall-sit (isometric): ทำไปกี่วิ + ผิดอะไรบ้าง
        let wallSitErrors: [ErrorCount] = wallSitErrorCounts
            .filter { $0.value > 0 }
            .map { ErrorCount(reason: Self.displayLabel(for: $0.key), count: $0.value) }
            .sorted { $0.count > $1.count }
        items.append(.isometric(
            name: "Wall-Sit",
            durationSeconds: correctSeconds,
            targetSeconds: targetSeconds,
            errors: wallSitErrors
        ))

        // Squat (movement): ทำทั้งหมดกี่ครั้ง · เป้าหมาย + ผิดอะไรบ้าง
        let squatErrors: [ErrorCount] = incorrectReps > 0
            ? [ErrorCount(reason: "Knees in", count: incorrectReps)]
            : []
        items.append(.movement(
            name: "Squat",
            totalReps: totalReps,
            correctReps: correctReps,
            incorrectReps: incorrectReps,
            targetCorrectReps: squatTargetCorrectReps,
            errors: squatErrors
        ))

        // Plank (isometric)
        let plankErrors: [ErrorCount] = plankErrorCounts
            .filter { $0.value > 0 }
            .map { ErrorCount(reason: Self.displayLabel(for: $0.key), count: $0.value) }
            .sorted { $0.count > $1.count }
        items.append(.isometric(
            name: "Plank",
            durationSeconds: plankCorrectSeconds,
            targetSeconds: plankTargetSeconds,
            errors: plankErrors
        ))

        return SessionSummary(items: items, totalTimeSeconds: totalTime, estimatedCalories: calories)
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

        // รอ 2.5 วิ หลัง Stand OK
        if Date().timeIntervalSince(okAt) > 2.5 {
            // กันพูดรัว (พูดทุก ~3 วิ)
            if Date().timeIntervalSince(lastPleaseStartAt) > 3.0 {
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
        squatPreviewSeconds = 5

        // countdown UI
        for i in 1...5 {
            DispatchQueue.main.asyncAfter(deadline: .now() + Double(i)) { [weak self] in
                guard let self else { return }
                if self.showSquatPreview {
                    self.squatPreviewSeconds = max(0, 5 - i)
                }
            }
        }

        DispatchQueue.main.asyncAfter(deadline: .now() + 5.0) { [weak self] in
            guard let self else { return }
            self.showSquatPreview = false
            self.switchToSquat()
        }
    }

    private func switchToSquat() {
        guard isSessionRunning else { return }
        guard !didSwitchToSquat else { return }

        didSwitchToSquat = true
        mode = .squat
        backendState = BackendPhase.NO_POSE.rawValue
        setFeedbackIfChanged("Switching to Squat…")

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

        // กัน wall-sit timer สะสมต่อ
        wallSitCountingActive = false
        wallSitIsCorrectHold = false

        // start squat streaming
        cameraManager.stopStreaming()
        cameraManager.startStreaming(to: activeWSURL)

        speech.speak("Switch to squat", language: speechLang, minInterval: 0)
    }

    private func switchToPlank() {
        guard isSessionRunning else { return }
        guard !didSwitchToPlank else { return }

        didSwitchToPlank = true
        mode = .plank
        backendState = BackendPhase.NO_POSE.rawValue
        setFeedbackIfChanged("Switching to Plank…")

        // reset plank counters
        plankCorrectSeconds = 0
        plankTargetSeconds = 5.0
        plankConsecutiveCorrect = 0
        plankCountingActive = false
        plankIsCorrectHold = false
        passedPlank = false

        cameraManager.stopStreaming()
        cameraManager.startStreaming(to: activeWSURL)

        speech.speak("Switch to plank", language: speechLang, minInterval: 0)
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
                    // 1️⃣ STAND OK
                    if predRaw == "stand_ok" {
                        if !self.squatStandOK {
                            self.squatStandOK = true
                            self.lastStandOKAt = Date()
                            self.speech.speak("OK", language: self.speechLang, minInterval: 0)
                        }
                        return
                    }

                    // 2️⃣ START SQUAT (any non-stand phase)
                    if predRaw != "stand_ok" && predRaw != "stand" {
                        self.squatStarted = true
                    }

                    // 3️⃣ REP UPDATE (backend ส่ง total, dataset_correct/good, incorrect)
                    if let reps = obj["reps"] as? [String: Any] {
                        if let total = reps["total"] as? Int {
                            self.totalReps = total
                        }

                        let correct: Int? = (reps["correct"] as? Int)
                            ?? (reps["dataset_correct"] as? Int)
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
                            self.speech.speak("Squat completed", language: self.speechLang, minInterval: 0)
                            self.switchToPlank()
                        }
                    }
                    return
                }

                // PLANK
                if self.mode == .plank {
                    self.handlePlankResult(pred: predRaw, conf: conf)
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

        // นับข้อผิดพลาดแต่ละประเภท (สำหรับ summary)
        if !isCorrectNow {
            wallSitErrorCounts[pred] = (wallSitErrorCounts[pred] ?? 0) + 1
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
                    setFeedbackIfChanged("Reset • Try again")
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
            setFeedbackIfChanged("Reset • Hold correct again")
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

        let isCorrectNow = p.contains("correct") && c >= wallSitConfThreshold

        plankIsCorrectHold = isCorrectNow

        if !isCorrectNow {
            plankErrorCounts[pred] = (plankErrorCounts[pred] ?? 0) + 1
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
                    setFeedbackIfChanged("Reset • Try plank again")
                }
            }
            return
        }

        if !isCorrectNow {
            plankCountingActive = false
            plankIsCorrectHold = false
            plankConsecutiveCorrect = 0
            plankCorrectSeconds = 0
            setFeedbackIfChanged("Reset • Hold plank correct again")
        }
    }

    // MARK: - Helpers
    private func setFeedbackIfChanged(_ newText: String) {
        if feedback != newText {
            feedback = newText
        }
    }
}

