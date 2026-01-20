import SwiftUI
import AVFoundation

// MARK: - SpeechManager
/// A small text-to-speech helper that:
/// - configures `AVAudioSession` for spoken feedback
/// - cleans backend text into a speakable phrase
/// - rate-limits and de-duplicates speech to avoid spamming
final class SpeechManager: ObservableObject {

    private let speaker = AVSpeechSynthesizer()
    private var lastSpoken: String = ""
    private var lastSpokenAt: Date = .distantPast

    /// Configure audio session so speech is audible in most scenarios.
    func configureAudioSession() {
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playback, mode: .spokenAudio, options: [.duckOthers])
            try session.setActive(true)
        } catch {
            print("Audio session error: \(error)")
        }
    }

    /// Clean raw feedback into a short phrase suitable for TTS.
    /// Removes confidence values, symbols, and converts underscores to spaces.
    func cleanForSpeech(_ text: String) -> String {
        var s = text.trimmingCharacters(in: .whitespacesAndNewlines)

        if let dot = s.firstIndex(of: "•") { s = String(s[..<dot]) }
        if let paren = s.firstIndex(of: "(") { s = String(s[..<paren]) }

        s = s.replacingOccurrences(
            of: #"[\d\.\%\-\+]+"#,
            with: "",
            options: .regularExpression
        )

        s = s.replacingOccurrences(of: "_", with: " ")

        s = s.replacingOccurrences(
            of: #"[^A-Za-zก-๙\s]+"#,
            with: "",
            options: .regularExpression
        )

        s = s.replacingOccurrences(
            of: #"\s+"#,
            with: " ",
            options: .regularExpression
        )

        return s.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Decide whether the cleaned text should be spoken.
    func shouldSpeak(_ rawText: String) -> Bool {
        let t = rawText.lowercased()
        if t.isEmpty { return false }
        if t.contains("correct") { return false }
        if t.contains("hold") { return false }
        return true
    }

    /// Speak with de-duplication and rate limiting.
    func speak(
        _ text: String,
        language: String = "en-US",
        minInterval: TimeInterval = 1.2,
        allowRepeat: Bool = false
    ) {
        let clean = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !clean.isEmpty else { return }

        if !allowRepeat, clean == lastSpoken { return }

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

// MARK: - WorkoutSessionView
/// Runs Wall-sit first, then shows a short Squat preview countdown and auto-switches to Squat.
/// - Wall-sit: frontend counts 5 seconds of correct hold
/// - Squat: backend provides rep counts via `reps { total, correct, incorrect }`
struct WorkoutSessionView: View {

    enum Mode { case wallSit, squat }

    let setTitle: String

    @ObservedObject var cameraManager = CameraManager()

    @State private var feedback: String = "Side view • Stand → Lower → Hold"
    @State private var totalReps = 0
    @State private var correctReps = 0
    @State private var incorrectReps = 0
    @State private var startTime = Date()
    @State private var navigateToResult = false
    @State private var isSessionRunning = false
    @State private var backendState: String = "waiting"

    @State private var squatStandOK: Bool = false
    @State private var squatStarted: Bool = false
    @State private var lastStandOKAt: Date?
    @State private var lastPleaseStartAt: Date = .distantPast
    @State private var lastSpokenCorrectReps: Int = 0

    @State private var correctSeconds: Double = 0
    @State private var targetSeconds: Double = 5.0
    @State private var passedWallSit: Bool = false

    @State private var wallSitConsecutiveCorrect: Int = 0
    @State private var wallSitCountingActive: Bool = false
    @State private var wallSitIsCorrectHold: Bool = false

    private let wallSitTick = Timer.publish(every: 0.1, on: .main, in: .common).autoconnect()
    private let squatIdleTick = Timer.publish(every: 0.5, on: .main, in: .common).autoconnect()

    @State private var showSquatPreview: Bool = false
    @State private var squatPreviewSeconds: Int = 5
    @State private var didSwitchToSquat: Bool = false

    @State private var squatTargetCorrectReps: Int = 3
    @State private var mode: Mode = .wallSit

    @StateObject private var speech = SpeechManager()
    private let speechLang = "en-US"

    private let wsWallSitURL = "ws://172.20.10.5:5050/ws/video"
    private let wsSquatURL   = "ws://172.20.10.5:5051/ws/video"

    private var activeWSURL: String {
        mode == .wallSit ? wsWallSitURL : wsSquatURL
    }

    private let wallSitConfThreshold: Double = 0.50
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

            if !isSessionRunning {
                guideCenterOverlay
                    .transition(.opacity)
            }

            if showSquatPreview {
                squatPreviewOverlay
                    .transition(.opacity)
            }

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
        .onReceive(wallSitTick) { _ in
            guard isSessionRunning else { return }
            guard mode == .wallSit else { return }
            guard wallSitCountingActive else { return }
            guard !passedWallSit else { return }
            guard wallSitIsCorrectHold else { return }
            guard !showSquatPreview else { return }

            correctSeconds = min(targetSeconds, correctSeconds + 0.1)

            if correctSeconds >= targetSeconds {
                correctSeconds = targetSeconds
                passedWallSit = true
                setFeedbackIfChanged("Passed")

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

            if Date().timeIntervalSince(okAt) > 2.5 {
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
        if passedWallSit { return showSquatPreview ? "PASSED (Next: Squat)" : "PASSED (Switching…)" }
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

    private func startSession() {
        withAnimation(.easeInOut(duration: 0.2)) {
            isSessionRunning = true
        }

        startTime = Date()
        backendState = "starting"
        feedback = "Streaming to backend..."

        mode = .wallSit
        didSwitchToSquat = false
        passedWallSit = false

        showSquatPreview = false
        squatPreviewSeconds = 5

        totalReps = 0
        correctReps = 0
        incorrectReps = 0
        lastSpokenCorrectReps = 0

        correctSeconds = 0
        targetSeconds = 5.0
        wallSitConsecutiveCorrect = 0
        wallSitCountingActive = false
        wallSitIsCorrectHold = false

        cameraManager.startStreaming(to: activeWSURL)
        speech.speak("Session started", language: speechLang, minInterval: 0)
    }

    private func stopSession() {
        isSessionRunning = false
        cameraManager.stopStreaming()
        backendState = "waiting"
        speech.stop()
    }

    private func finishSession() {
        stopSession()
        navigateToResult = true
    }

    private func startSquatPreviewThenSwitch() {
        showSquatPreview = true
        squatPreviewSeconds = 5

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
        backendState = "starting"
        setFeedbackIfChanged("Switching to Squat…")

        totalReps = 0
        correctReps = 0
        incorrectReps = 0
        lastSpokenCorrectReps = 0

        wallSitCountingActive = false
        wallSitIsCorrectHold = false

        cameraManager.stopStreaming()
        cameraManager.startStreaming(to: activeWSURL)

        speech.speak("Switch to squat", language: speechLang, minInterval: 0)
    }

    private func handleBackendMessage(_ text: String) {
        guard let data = text.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return
        }

        let type = obj["type"] as? String ?? ""

        if type == "result" {
            let pred = (obj["prediction"] as? String ?? "...")
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .lowercased()

            let conf = obj["confidence"] as? Double

            DispatchQueue.main.async {
                if let conf {
                    self.setFeedbackIfChanged("\(pred) • \(String(format: "%.2f", conf))")
                } else {
                    self.setFeedbackIfChanged("\(pred)")
                }

                if self.mode == .wallSit {
                    self.handleWallSitResult(pred: pred, conf: conf)
                    return
                }

                if self.mode == .squat {

                    if pred == "stand_ok" {
                        if !self.squatStandOK {
                            self.squatStandOK = true
                            self.lastStandOKAt = Date()
                            self.speech.speak("OK", language: self.speechLang, minInterval: 0)
                        }
                        return
                    }

                    if pred != "stand_ok" && pred != "stand" {
                        self.squatStarted = true
                    }

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

        if type == "status" {
            let state = obj["state"] as? String ?? "..."
            let msg = obj["message"] as? String
            DispatchQueue.main.async {
                self.backendState = state
                if let msg { self.setFeedbackIfChanged(msg) }
            }
            return
        }

        if type == "info" {
            let msg = obj["message"] as? String ?? "..."
            DispatchQueue.main.async { self.setFeedbackIfChanged(msg) }
            return
        }
    }

    private func handleWallSitResult(pred: String, conf: Double?) {
        let c = conf ?? 0.0
        let isCorrectNow = (pred == "correct") && (c >= wallSitConfThreshold)

        wallSitIsCorrectHold = isCorrectNow

        if passedWallSit { return }

        if !wallSitCountingActive {
            if isCorrectNow {
                wallSitCountingActive = true
                correctSeconds = 0
                targetSeconds = 5.0
                speech.speak("OK", language: speechLang, minInterval: 0)
                setFeedbackIfChanged("Start hold…")
            }
            return
        }

        if !isCorrectNow {
            wallSitCountingActive = false
            wallSitIsCorrectHold = false
            correctSeconds = 0
            setFeedbackIfChanged("Reset • Hold correct again")
        }
    }

    private func setFeedbackIfChanged(_ newText: String) {
        if feedback != newText {
            feedback = newText
        }
    }
}

// MARK: - TopHUD
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

// MARK: - WallSitGuideOverlayCompact
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

// MARK: - SquatGuideOverlayCompact
private struct SquatGuideOverlayCompact: View {
    private let frames = ["squat_01", "squat_02", "squat_03"]
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

// MARK: - Progress Bar
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
