import SwiftUI
import AVFoundation

// MARK: - Speech Manager (คงอยู่ ไม่โดน recreate)
final class SpeechManager: ObservableObject {
    private let speaker = AVSpeechSynthesizer()
    private var lastSpoken: String = ""
    private var lastSpokenAt: Date = .distantPast

    func configureAudioSession() {
        do {
            let session = AVAudioSession.sharedInstance()
            // playback + spokenAudio ช่วยให้เสียงออกแม้ silent หลายกรณี
            try session.setCategory(.playback, mode: .spokenAudio, options: [.duckOthers])
            try session.setActive(true)
        } catch {
            print("Audio session error: \(error)")
        }
    }

    // ✅ ล้างข้อความ: เอาตัวเลข/สัญลักษณ์ confidence ออก
    func cleanForSpeech(_ text: String) -> String {
        var s = text.trimmingCharacters(in: .whitespacesAndNewlines)

        // ตัดหลัง "•" (เช่น "knees_in • 0.92")
        if let dot = s.firstIndex(of: "•") {
            s = String(s[..<dot])
        }

        // ตัดหลัง "(" (เช่น "knees_in (0.923)")
        if let paren = s.firstIndex(of: "(") {
            s = String(s[..<paren])
        }

        // เอาตัวเลข/จุด/เปอร์เซ็นต์ ออก
        s = s.replacingOccurrences(
            of: #"[\d\.\%\-\+]+"#,
            with: "",
            options: .regularExpression
        )

        // ปรับ underscore -> เว้นวรรค
        s = s.replacingOccurrences(of: "_", with: " ")

        // เก็บเฉพาะตัวอักษร + เว้นวรรค (รองรับไทย/อังกฤษ)
        s = s.replacingOccurrences(
            of: #"[^A-Za-zก-๙\s]+"#,
            with: "",
            options: .regularExpression
        )

        // จัดช่องว่างซ้ำ
        s = s.replacingOccurrences(
            of: #"\s+"#,
            with: " ",
            options: .regularExpression
        )

        return s.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // ✅ เงื่อนไขว่าจะพูดไหม (กันรัว)
    func shouldSpeak(_ rawText: String) -> Bool {
        let t = rawText.lowercased()
        if t.isEmpty { return false }
        if t.contains("correct") { return false } // ไม่พูดถ้า correct
        if t.contains("hold") { return false }    // กันพูด "hold..." รัว
        // "passed" ให้พูดได้
        return true
    }

    func speak(
        _ text: String,
        language: String = "en-US",
        minInterval: TimeInterval = 1.2,
        allowRepeat: Bool = false
    ) {
        let clean = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !clean.isEmpty else { return }

        // ✅ กันพูดซ้ำ (ถ้าไม่ได้ allowRepeat)
        if !allowRepeat, clean == lastSpoken { return }

        // กันพูดถี่เกิน (ถ้า minInterval > 0)
        let now = Date()
        if minInterval > 0, now.timeIntervalSince(lastSpokenAt) < minInterval { return }

        lastSpoken = clean
        lastSpokenAt = now

        let u = AVSpeechUtterance(string: clean)
        u.voice = AVSpeechSynthesisVoice(language: language)
        u.rate = 0.50
        u.pitchMultiplier = 1.0
        u.volume = 1.0

        speaker.stopSpeaking(at: .immediate)
        speaker.speak(u)
    }

    func stop() {
        speaker.stopSpeaking(at: .immediate)
    }
}

// MARK: - Workout Session (Wall-sit -> show Squat preview 5s -> auto switch to Squat)
struct WorkoutSessionView: View {
    enum Mode { case wallSit, squat }

    // ✅ Backend phases (match Python)
    private enum BackendPhase: String {
        case NO_POSE, HAVE_POSE, BUFFERING, INFERENCING
    }

    let setTitle: String

    @ObservedObject var cameraManager = CameraManager()

    @State private var feedback: String = "Side view • Stand → Lower → Hold"
    @State private var totalReps = 0
    @State private var correctReps = 0
    @State private var incorrectReps = 0
    @State private var startTime = Date()
    @State private var navigateToResult = false
    @State private var isSessionRunning = false

    // ✅ show backend phase on HUD
    @State private var backendState: String = BackendPhase.NO_POSE.rawValue

    // ✅ Squat speech control
    @State private var squatStandOK: Bool = false
    @State private var squatStarted: Bool = false
    @State private var lastStandOKAt: Date?
    @State private var lastPleaseStartAt: Date = .distantPast

    @State private var lastSpokenCorrectReps: Int = 0

    // ✅ Wall-sit hold progress (5s pass) - นับใน Frontend
    @State private var correctSeconds: Double = 0
    @State private var targetSeconds: Double = 5.0
    @State private var passedWallSit: Bool = false

    // ✅ Gate: ต้อง correct 3 ครั้งติดก่อนถึงเริ่มนับเวลา
    @State private var wallSitConsecutiveCorrect: Int = 0
    @State private var wallSitCountingActive: Bool = false
    @State private var wallSitIsCorrectHold: Bool = false   // ✅ สำคัญ: ให้ Timer นับได้แม้ backend DEDUP
    private let wallSitTick = Timer.publish(every: 0.1, on: .main, in: .common).autoconnect()

    // ✅ Squat idle timer (ยืนเฉยหลัง Stand OK)
    private let squatIdleTick = Timer.publish(every: 0.5, on: .main, in: .common).autoconnect()

    // ✅ Preview Squat 5s
    @State private var showSquatPreview: Bool = false
    @State private var squatPreviewSeconds: Int = 5

    // ✅ Auto switch control
    @State private var didSwitchToSquat: Bool = false

    // ✅ Squat target: correct 3 reps = finish
    @State private var squatTargetCorrectReps: Int = 3

    // ✅ Current mode (เริ่มจาก wall-sit)
    @State private var mode: Mode = .wallSit

    // ✅ Speech manager
    @StateObject private var speech = SpeechManager()
    private let speechLang = "en-US"

    // ✅ WS endpoints
    private let wsWallSitURL = "ws://172.20.10.5:5050/ws/video"
    private let wsSquatURL   = "ws://172.20.10.5:5051/ws/video"

    private var activeWSURL: String {
        mode == .wallSit ? wsWallSitURL : wsSquatURL
    }

    // ปรับ threshold ตามโมเดลจริงของคุณ
    private let wallSitConfThreshold: Double = 0.50

    // เปิด/ปิดปุ่มทดสอบเสียง
    private let showTestVoiceButton = false

    private var wallSitProgress01: Double {
        guard targetSeconds > 0 else { return 0 }
        return min(1.0, max(0.0, correctSeconds / targetSeconds))
    }

    private var squatProgress01: Double {
        let tgt = Double(max(1, squatTargetCorrectReps))
        return min(1.0, max(0.0, Double(correctReps) / tgt))
    }

    var body: some View {
        ZStack {
            // ✅ Full-screen camera background
            CameraPreviewView(session: cameraManager.session)
                .ignoresSafeArea()

            Color.black.opacity(0.06).ignoresSafeArea()

            VStack {
                TopHUD(
                    isSessionRunning: isSessionRunning,
                    backendState: backendState,
                    setTitle: titleForHUD,
                    feedback: feedback
                )
                .padding(.horizontal, 14)
                .padding(.top, 8)

                Spacer()

                bottomPanel
                    .padding(.horizontal, 14)
                    .padding(.bottom, 10)
            }

            // ✅ Guide overlay (before start)
            if !isSessionRunning {
                guideCenterOverlay
                    .transition(.opacity)
            }

            // ✅ Squat preview overlay (5s)
            if showSquatPreview {
                squatPreviewOverlay
                    .transition(.opacity)
            }

            // ✅ Navigation trigger to Result
            NavigationLink(
                destination: WorkoutResultView(
                    totalReps: totalReps,
                    correctReps: correctReps,
                    incorrectReps: incorrectReps,
                    totalTime: Int(Date().timeIntervalSince(startTime)),
                    estimatedCalories: max(0, totalReps) * 4
                )
                .navigationBarBackButtonHidden(true),
                isActive: $navigateToResult
            ) { EmptyView() }
                .hidden()
        }
        .toolbar(.hidden, for: .navigationBar)
        .onAppear {
            speech.configureAudioSession()

            cameraManager.startSession()
            cameraManager.onBackendMessage = { text in
                handleBackendMessage(text)
            }
        }
        .onDisappear {
            stopSession()
            cameraManager.stopSession()
        }
        .onChange(of: feedback) { newValue in
            guard isSessionRunning else { return }
            let cleaned = speech.cleanForSpeech(newValue)
            guard speech.shouldSpeak(cleaned) else { return }
            speech.speak(cleaned, language: speechLang, minInterval: 1.2)
        }
        // ✅ นับเวลาเองตอน Hold (แก้ปัญหา backend DEDUP ส่ง result ไม่ถี่)
        .onReceive(wallSitTick) { _ in
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
        .onReceive(squatIdleTick) { _ in
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
    }

    private var titleForHUD: String {
        switch mode {
        case .wallSit: return "\(setTitle) • Wall-Sit"
        case .squat:   return "\(setTitle) • Squat"
        }
    }

    // MARK: - Center Guide
    private var guideCenterOverlay: some View {
        VStack(spacing: 12) {
            WallSitGuideOverlayCompact()

            Text("Set camera to SIDE VIEW\nPress Start when ready")
                .font(.subheadline.weight(.semibold))
                .foregroundColor(.white)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 16)
                .padding(.vertical, 12)
                .background(Color.black.opacity(0.45))
                .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
        }
        .padding(.horizontal, 18)
    }

    // MARK: - Squat Preview Overlay
    private var squatPreviewOverlay: some View {
        ZStack {
            Color.black.opacity(0.45).ignoresSafeArea()

            VStack(spacing: 12) {
                SquatGuideOverlayCompact()

                Text("Next: Squat\nStarting in \(squatPreviewSeconds)s")
                    .font(.subheadline.weight(.semibold))
                    .foregroundColor(.white)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 12)
                    .background(Color.black.opacity(0.45))
                    .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
            }
            .padding(.horizontal, 18)
        }
    }

    // MARK: - Bottom Panel
    private var bottomPanel: some View {
        VStack(spacing: 10) {
            HStack(spacing: 10) {
                statChip(title: "Total", value: totalReps, tint: .white)
                statChip(title: "Correct", value: correctReps, tint: .green)
                statChip(title: "Incorrect", value: incorrectReps, tint: .red)
            }

            progressCard

            if let err = cameraManager.lastError {
                Text("Error: \(err)")
                    .font(.caption)
                    .foregroundColor(.red)
                    .padding(.top, 2)
            }

            if showTestVoiceButton {
                Button("Test Voice") {
                    speech.speak("This is a voice test", language: speechLang, minInterval: 0)
                }
                .padding(.vertical, 10)
                .frame(maxWidth: .infinity)
                .background(Color.black.opacity(0.25))
                .foregroundColor(.white)
                .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
            }

            Button {
                if isSessionRunning {
                    stopSession()
                    navigateToResult = true
                } else {
                    startSession()
                }
            } label: {
                Text(isSessionRunning ? "End Session" : "Start Session")
                    .font(.headline)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
                    .foregroundColor(.black)
                    .background(isSessionRunning ? Color.red : Color.green)
                    .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
            }
        }
    }

    private var progressCard: some View {
        Group {
            if mode == .wallSit {
                wallSitProgressCard
            } else {
                squatProgressCard
            }
        }
    }

    private var wallSitProgressCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(wallSitTitleText)
                    .font(.caption.weight(.semibold))
                    .foregroundColor(.white)
                Spacer()

                if wallSitCountingActive {
                    Text("\(String(format: "%.1f", correctSeconds)) / \(String(format: "%.1f", targetSeconds)) s")
                        .font(.caption2.weight(.bold))
                        .foregroundColor(.white.opacity(0.9))
                        .monospacedDigit()
                } else {
                    Text("\(wallSitConsecutiveCorrect) / 3 correct")
                        .font(.caption2.weight(.bold))
                        .foregroundColor(.white.opacity(0.9))
                        .monospacedDigit()
                }
            }

            progressBar(wallSitProgress01, height: 10)
        }
        .padding(.vertical, 10)
        .padding(.horizontal, 12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.black.opacity(0.24))
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
    }

    private var wallSitTitleText: String {
        if passedWallSit { return showSquatPreview ? "PASSED ✅ (Next: Squat)" : "PASSED ✅ (Switching…)" }
        if wallSitCountingActive { return "Hold Correct (5s)" }
        return "Get 3x Correct to start"
    }

    private var squatProgressCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Squat Goal")
                    .font(.caption.weight(.semibold))
                    .foregroundColor(.white)
                Spacer()
                Text("\(correctReps) / \(squatTargetCorrectReps) correct")
                    .font(.caption2.weight(.bold))
                    .foregroundColor(.white.opacity(0.9))
                    .monospacedDigit()
            }

            progressBar(squatProgress01, height: 10)
        }
        .padding(.vertical, 10)
        .padding(.horizontal, 12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.black.opacity(0.24))
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
    }

    private func statChip(title: String, value: Int, tint: Color) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption2)
                .foregroundColor(.white.opacity(0.7))
            Text("\(value)")
                .font(.system(size: 22, weight: .bold, design: .rounded))
                .monospacedDigit()
                .foregroundColor(tint)
        }
        .padding(.vertical, 10)
        .padding(.horizontal, 12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.black.opacity(0.24))
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
    }

    // MARK: - Session Control
    private func startSession() {
        withAnimation(.easeInOut(duration: 0.2)) {
            isSessionRunning = true
        }

        startTime = Date()
        backendState = BackendPhase.NO_POSE.rawValue
        feedback = "Streaming to backend..."

        // ✅ reset ALL
        mode = .wallSit
        didSwitchToSquat = false
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

        // squat helpers
        squatStandOK = false
        squatStarted = false
        lastStandOKAt = nil
        lastPleaseStartAt = .distantPast

        cameraManager.startStreaming(to: activeWSURL)
        speech.speak("Session started", language: speechLang, minInterval: 0)
    }

    private func stopSession() {
        isSessionRunning = false
        cameraManager.stopStreaming()
        backendState = BackendPhase.NO_POSE.rawValue
        speech.stop()
    }

    private func finishSession() {
        stopSession()
        navigateToResult = true
    }

    private func startSquatPreviewThenSwitch() {
        showSquatPreview = true
        squatPreviewSeconds = 5

        // countdown UI
        for i in 1...5 {
            DispatchQueue.main.asyncAfter(deadline: .now() + Double(i)) {
                if self.showSquatPreview {
                    self.squatPreviewSeconds = max(0, 5 - i)
                }
            }
        }

        DispatchQueue.main.asyncAfter(deadline: .now() + 5.0) {
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

        // ✅ reset squat counters (เริ่มนับใหม่)
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

                // ✅ IMPORTANT: ถ้า NO_POSE ให้ reset hold/ready ต่าง ๆ (กัน timer นับค้าง)
                if stateRaw == BackendPhase.NO_POSE.rawValue {
                    if self.mode == .wallSit {
                        self.wallSitIsCorrectHold = false
                        self.wallSitCountingActive = false
                        self.wallSitConsecutiveCorrect = 0
                        self.correctSeconds = 0
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

                // ✅ WALL-SIT
                if self.mode == .wallSit {
                    self.handleWallSitResult(pred: predRaw, conf: conf)
                    return
                }

                // ✅ SQUAT
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

                    // 3️⃣ REP UPDATE
                    if let reps = obj["reps"] as? [String: Any] {
                        if let total = reps["total"] as? Int {
                            self.totalReps = total
                        }

                        if let correct = reps["correct"] as? Int {
                            if correct > self.correctReps {
                                self.speech.speak("Good", language: self.speechLang, minInterval: 0.5, allowRepeat: true)
                                self.lastSpokenCorrectReps = correct
                            }
                            self.correctReps = correct
                        }

                        if let incorrect = reps["incorrect"] as? Int {
                            self.incorrectReps = incorrect
                        }

                        if self.correctReps >= self.squatTargetCorrectReps {
                            self.speech.speak("Completed", language: self.speechLang, minInterval: 0)
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
        // ✅ รองรับ label แบบ dataset_correct / correct / passed ฯลฯ
        let p = pred.lowercased()
        let c = conf ?? 0.0

        // ถือว่า "correct" ถ้า label มีคำว่า correct และ confidence ผ่าน threshold
        // (เช่น dataset_correct)
        let isCorrectNow = p.contains("correct") && c >= wallSitConfThreshold

        // ให้ timer รู้ว่าตอนนี้ correct อยู่ไหม (แม้ backend dedup)
        wallSitIsCorrectHold = isCorrectNow

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

    // MARK: - Helpers
    private func setFeedbackIfChanged(_ newText: String) {
        if feedback != newText {
            feedback = newText
        }
    }
}

// MARK: - Top HUD
private struct TopHUD: View {
    let isSessionRunning: Bool
    let backendState: String
    let setTitle: String
    let feedback: String

    var body: some View {
        VStack(spacing: 6) {
            HStack(spacing: 8) {
                pill(text: isSessionRunning ? "Streaming" : "Preview",
                     dot: isSessionRunning ? .green : .gray)
                statePill(backendState)
                Spacer()
                Text(setTitle)
                    .font(.caption.weight(.semibold))
                    .foregroundColor(.white.opacity(0.95))
                    .lineLimit(1)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 7)
                    .background(Color.black.opacity(0.30))
                    .clipShape(Capsule())
            }

            HStack(spacing: 8) {
                Text("Live")
                    .font(.caption2.weight(.bold))
                    .foregroundColor(.white.opacity(0.9))
                    .padding(.horizontal, 10)
                    .padding(.vertical, 7)
                    .background(Color.black.opacity(0.28))
                    .clipShape(Capsule())

                Text(feedback)
                    .font(.caption.weight(.semibold))
                    .foregroundColor(.white)
                    .lineLimit(1)
                    .minimumScaleFactor(0.75)

                Spacer(minLength: 0)
            }
        }
        .padding(.vertical, 10)
        .padding(.horizontal, 12)
        .background(Color.black.opacity(0.22))
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
    }

    private func pill(text: String, dot: Color) -> some View {
        HStack(spacing: 6) {
            Circle().fill(dot).frame(width: 8, height: 8)
            Text(text).font(.caption.weight(.semibold)).foregroundColor(.white)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 7)
        .background(Color.black.opacity(0.32))
        .clipShape(Capsule())
    }

    private func statePill(_ state: String) -> some View {
        Text(state.uppercased())
            .font(.caption2.weight(.bold))
            .foregroundColor(.white.opacity(0.95))
            .padding(.horizontal, 10)
            .padding(.vertical, 7)
            .background(Color.black.opacity(0.28))
            .clipShape(Capsule())
    }
}

// MARK: - Guide Overlay Compact (Wall-Sit)
private struct WallSitGuideOverlayCompact: View {
    private let frames = ["wall_01", "wall_02", "wall_03"]
    @State private var idx: Int = 0
    private let timer = Timer.publish(every: 1.1, on: .main, in: .common).autoconnect()

    var body: some View {
        VStack(spacing: 10) {
            Image(frames[idx])
                .resizable()
                .scaledToFit()
                .frame(maxWidth: 260, maxHeight: 200)
                .shadow(radius: 10)

            Text("Stand → Lower → Hold")
                .font(.caption.weight(.semibold))
                .foregroundColor(.white)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Color.black.opacity(0.45))
                .clipShape(Capsule())
        }
        .onReceive(timer) { _ in
            withAnimation(.easeInOut(duration: 0.2)) {
                idx = (idx + 1) % frames.count
            }
        }
    }
}

// MARK: - Guide Overlay Compact (Squat)
private struct SquatGuideOverlayCompact: View {
    private let frames = ["squat_01", "squat_02", "squat_03"] // ✅ ใส่รูปใน Assets
    @State private var idx: Int = 0
    private let timer = Timer.publish(every: 1.1, on: .main, in: .common).autoconnect()

    var body: some View {
        VStack(spacing: 10) {
            Image(frames[idx])
                .resizable()
                .scaledToFit()
                .frame(maxWidth: 260, maxHeight: 200)
                .shadow(radius: 10)

            Text("Squat • Chest up • Knees out")
                .font(.caption.weight(.semibold))
                .foregroundColor(.white)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Color.black.opacity(0.45))
                .clipShape(Capsule())
        }
        .onReceive(timer) { _ in
            withAnimation(.easeInOut(duration: 0.2)) {
                idx = (idx + 1) % frames.count
            }
        }
    }
}

// MARK: - Custom Progress Bar
private func progressBar(_ progress: Double, height: CGFloat = 10) -> some View {
    GeometryReader { geo in
        ZStack(alignment: .leading) {
            RoundedRectangle(cornerRadius: height)
                .fill(Color.white.opacity(0.25))

            RoundedRectangle(cornerRadius: height)
                .fill(progress >= 1.0 ? Color.green : Color.white)
                .frame(width: max(0, geo.size.width * CGFloat(min(progress, 1.0))))
                .animation(.easeInOut(duration: 0.25), value: progress)
        }
    }
    .frame(height: height)
}
