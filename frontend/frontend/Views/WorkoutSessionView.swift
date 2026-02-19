import SwiftUI

// MARK: - Workout Session (Wall-sit -> show Squat preview 5s -> auto switch to Squat)
struct WorkoutSessionView: View {
    let setTitle: String

    @StateObject private var viewModel: WorkoutSessionViewModel

    // ✅ Wall-sit hold timer
    private let wallSitTick = Timer.publish(every: 0.1, on: .main, in: .common).autoconnect()

    // ✅ Squat idle timer (ยืนเฉยหลัง Stand OK)
    private let squatIdleTick = Timer.publish(every: 0.5, on: .main, in: .common).autoconnect()

    // ✅ Plank hold timer
    private let plankTick = Timer.publish(every: 0.1, on: .main, in: .common).autoconnect()

    init(setTitle: String) {
        self.setTitle = setTitle
        _viewModel = StateObject(wrappedValue: WorkoutSessionViewModel(setTitle: setTitle))
    }

    var body: some View {
        NavigationStack {
            ZStack {
                CameraPreviewView(session: viewModel.cameraManager.session)
                    .ignoresSafeArea()

                Color.black.opacity(0.06).ignoresSafeArea()

                VStack {
                    TopHUD(
                        isSessionRunning: viewModel.isSessionRunning,
                        backendState: viewModel.backendState,
                        setTitle: viewModel.titleForHUD,
                        feedback: viewModel.feedback
                    )
                    .padding(.horizontal, 14)
                    .padding(.top, 8)

                    Spacer()

                    bottomPanel
                        .padding(.horizontal, 14)
                        .padding(.bottom, 10)
                }

                if !viewModel.isSessionRunning {
                    guideCenterOverlay
                        .transition(.opacity)
                }

                if viewModel.showSquatPreview {
                    squatPreviewOverlay
                        .transition(.opacity)
                }

                if viewModel.showPlankPreview {
                    plankPreviewOverlay
                        .transition(.opacity)
                }

                if showLightAdjustOverlay {
                    lightAdjustOverlay
                        .transition(.opacity)
                }
            }
            .navigationBarBackButtonHidden(true)
            .navigationDestination(isPresented: $viewModel.navigateToResult) {
                WorkoutResultView(summary: viewModel.sessionSummary)
                    .navigationBarBackButtonHidden(true)
            }
        }
        .toolbar(.hidden, for: .navigationBar)
        .onAppear {
            viewModel.onAppear()
        }
        .onDisappear {
            viewModel.onDisappear()
        }
        .onChange(of: viewModel.feedback) { oldValue, newValue in
            viewModel.handleFeedbackChange()
        }
        // ✅ นับเวลาเองตอน Hold (แก้ปัญหา backend DEDUP ส่ง result ไม่ถี่)
        .onReceive(wallSitTick) { _ in
            viewModel.handleWallSitTick()
        }
        .onReceive(squatIdleTick) { _ in
            viewModel.handleSquatIdleTick()
        }
        .onReceive(plankTick) { _ in
            viewModel.handlePlankTick()
        }
    }

    // MARK: - Center Guide (first exercise is wall-sit)
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
        .offset(y: -40)
    }

    // MARK: - Squat Preview Overlay
    private var squatPreviewOverlay: some View {
        ZStack {
            Color.black.opacity(0.45).ignoresSafeArea()

            VStack(spacing: 12) {
                SquatGuideOverlayCompact()

                Text("Next: Squat\nStarting in \(viewModel.squatPreviewSeconds)s")
                    .font(.subheadline.weight(.semibold))
                    .foregroundColor(.white)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 12)
                    .background(Color.black.opacity(0.45))
                    .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
            }
            .padding(.horizontal, 18)
            .offset(y: -40)
        }
    }

    // MARK: - Plank Preview Overlay
    private var plankPreviewOverlay: some View {
        ZStack {
            Color.black.opacity(0.45).ignoresSafeArea()

            VStack(spacing: 12) {
                PlankGuideOverlayCompact()

                Text("Next: Plank\nStarting in \(viewModel.plankPreviewSeconds)s")
                    .font(.subheadline.weight(.semibold))
                    .foregroundColor(.white)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 12)
                    .background(Color.black.opacity(0.45))
                    .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
            }
            .padding(.horizontal, 18)
            .offset(y: -40)
        }
    }

    // MARK: - Light adjust overlay (เมื่อ backend เตือนว่าแสงน้อย)
    private var showLightAdjustOverlay: Bool {
        let f = viewModel.feedback.lowercased()
        return viewModel.isSessionRunning
            && (f.contains("adjust") && (f.contains("light") || f.contains("lights")) || f.contains("too dark") || f.contains("dark"))
    }

    private var lightAdjustOverlay: some View {
        ZStack {
            Color.black.opacity(0.45).ignoresSafeArea()

            VStack(spacing: 16) {
                Image("light")
                    .resizable()
                    .scaledToFit()
                    .frame(maxWidth: 280, maxHeight: 220)
                    .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))

                Text("Adjust your light")
                    .font(.subheadline.weight(.semibold))
                    .foregroundColor(.white)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 20)
                    .padding(.vertical, 12)
                    .background(Color.black.opacity(0.45))
                    .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
            }
            .padding(.horizontal, 24)
            .offset(y: -40)
        }
    }

    // MARK: - Bottom Panel
    private var bottomPanel: some View {
        VStack(spacing: 10) {
            HStack(spacing: 10) {
                statChip(title: "Total", value: viewModel.totalReps, tint: .white)
                statChip(title: "Correct", value: viewModel.correctReps, tint: .green)
                statChip(title: "Incorrect", value: viewModel.incorrectReps, tint: .red)
            }

            progressCard

            if let err = viewModel.cameraManager.lastError {
                Text("Error: \(err)")
                    .font(.caption)
                    .foregroundColor(.red)
                    .padding(.top, 2)
            }

            if viewModel.showTestVoiceButton {
                Button("Test Voice") {
                    viewModel.speech.speak("This is a voice test", language: "en-US", minInterval: 0)
                }
                .padding(.vertical, 10)
                .frame(maxWidth: .infinity)
                .background(Color.black.opacity(0.25))
                .foregroundColor(.white)
                .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
            }

            Button {
                if viewModel.isSessionRunning {
                    viewModel.finishSession()
                } else {
                    viewModel.startSession()
                }
            } label: {
                Text(viewModel.isSessionRunning ? "End Session" : "Start Session")
                    .font(.headline)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
                    .foregroundColor(.black)
                    .background(viewModel.isSessionRunning ? Color.red : Color.green)
                    .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
            }
        }
    }

    private var progressCard: some View {
        Group {
            if viewModel.mode == .wallSit {
                wallSitProgressCard
            } else if viewModel.mode == .squat {
                squatProgressCard
            } else {
                plankProgressCard
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

                if viewModel.wallSitCountingActive {
                    Text("\(String(format: "%.1f", viewModel.correctSeconds)) / \(String(format: "%.1f", viewModel.targetSeconds)) s")
                        .font(.caption2.weight(.bold))
                        .foregroundColor(.white.opacity(0.9))
                        .monospacedDigit()
                } else {
                    Text("\(viewModel.wallSitConsecutiveCorrect)")
                        .font(.caption2.weight(.bold))
                        .foregroundColor(.white.opacity(0.9))
                        .monospacedDigit()
                }
            }

            progressBar(viewModel.wallSitProgress01, height: 10)
        }
        .padding(.vertical, 10)
        .padding(.horizontal, 12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.black.opacity(0.24))
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
    }

    private var wallSitTitleText: String {
        if viewModel.passedWallSit { return viewModel.showSquatPreview ? "PASSED  (Next: Squat)" : "PASSED (Switching…)" }
        if viewModel.wallSitCountingActive { return "Hold Correct (5s)" }
        return "Get In Position"
    }

    private var plankTitleText: String {
        if viewModel.passedPlank { return "PASSED" }
        return "Plank Goal"
    }

    private var squatProgressCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Squat Goal")
                    .font(.caption.weight(.semibold))
                    .foregroundColor(.white)
                Spacer()
                Text("\(viewModel.correctReps) / \(viewModel.squatTargetCorrectReps) correct")
                    .font(.caption2.weight(.bold))
                    .foregroundColor(.white.opacity(0.9))
                    .monospacedDigit()
            }

            progressBar(viewModel.squatProgress01, height: 10)
        }
        .padding(.vertical, 10)
        .padding(.horizontal, 12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.black.opacity(0.24))
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
    }

    private var plankProgressCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(plankTitleText)
                    .font(.caption.weight(.semibold))
                    .foregroundColor(.white)
                Spacer()
                Text(String(format: "%.1f", viewModel.plankCorrectSeconds) + " / " + String(format: "%.1f", viewModel.plankTargetSeconds) + " s")
                    .font(.caption2.weight(.bold))
                    .foregroundColor(.white.opacity(0.9))
                    .monospacedDigit()
            }

            progressBar(viewModel.plankProgress01, height: 10)
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

// MARK: - Guide Overlay Compact (Wall-Sit) — Stand → Lower → Hold
private struct WallSitGuideOverlayCompact: View {
    private let frames = ["wall_01", "wall_02", "wall_03"]

    var body: some View {
        TimelineView(.periodic(from: Date(), by: 1.1)) { context in
            let idx = Int(context.date.timeIntervalSince1970 / 1.1) % frames.count
            VStack(spacing: 10) {
                Image(frames[idx])
                    .resizable()
                    .scaledToFit()
                    .frame(maxWidth: 260, maxHeight: 200)
                    .shadow(radius: 10)
                    .animation(.easeInOut(duration: 0.2), value: idx)

                Text("Stand → Lower → Hold")
                    .font(.caption.weight(.semibold))
                    .foregroundColor(.white)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(Color.black.opacity(0.45))
                    .clipShape(Capsule())
            }
        }
    }
}

// MARK: - Guide Overlay Compact (Squat) — ย่อ → หยุด → ขึ้น
private struct SquatGuideOverlayCompact: View {
    private let frames = ["squat_01", "squat_02", "squat_03"]

    var body: some View {
        TimelineView(.periodic(from: Date(), by: 1.1)) { context in
            let idx = Int(context.date.timeIntervalSince1970 / 1.1) % frames.count
            VStack(spacing: 10) {
                Image(frames[idx])
                    .resizable()
                    .scaledToFit()
                    .frame(maxWidth: 260, maxHeight: 200)
                    .shadow(radius: 10)
                    .animation(.easeInOut(duration: 0.2), value: idx)

                Text("Lower → Hold → Stand up")
                    .font(.caption.weight(.semibold))
                    .foregroundColor(.white)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(Color.black.opacity(0.45))
                    .clipShape(Capsule())
            }
        }
    }
}

// MARK: - Guide Overlay Compact (Plank)
private struct PlankGuideOverlayCompact: View {
    var body: some View {
        VStack(spacing: 10) {
            Image("plank_01")
                .resizable()
                .scaledToFit()
                .frame(maxWidth: 260, maxHeight: 200)
                .shadow(radius: 10)

            Text("Hold straight")
                .font(.caption.weight(.semibold))
                .foregroundColor(.white)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background(Color.black.opacity(0.45))
                .clipShape(Capsule())
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
