import SwiftUI

// MARK: - Workout Session View

/// Full-screen camera-based workout session with real-time backend feedback.
///
/// Displays a camera preview overlaid with a HUD (status pills, feedback text),
/// exercise guide overlays, progress cards, and exercise preview countdowns.
/// Timer publishers drive the isometric hold counters and idle detection.
struct WorkoutSessionView: View {
    let setTitle: String
    @Environment(\.dismiss) private var dismiss

    @StateObject private var viewModel: WorkoutSessionViewModel

    /// Wall-sit hold timer (0.1s interval for smooth progress bar updates).
    private let wallSitTick = Timer.publish(every: 0.1, on: .main, in: .common).autoconnect()

    /// Squat idle detection timer (checks if user is standing idle too long).
    private let squatIdleTick = Timer.publish(every: 0.5, on: .main, in: .common).autoconnect()

    /// Plank hold timer (0.1s interval).
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

                // Top and Bottom gradient protection for maximum text readability against bright camera feeds
                VStack {
                    LinearGradient(colors: [.black.opacity(0.8), .clear], startPoint: .top, endPoint: .bottom)
                        .frame(height: 160)
                    Spacer()
                    LinearGradient(colors: [.clear, .black.opacity(0.85)], startPoint: .top, endPoint: .bottom)
                        .frame(height: 380)
                }
                .ignoresSafeArea()

                VStack {
                    TopHUD(
                        isSessionRunning: viewModel.isSessionRunning,
                        setTitle: viewModel.titleForHUD
                    )
                    .padding(.horizontal, 16)
                    .padding(.top, 8)

                    Spacer()

                    // Massive Center Feedback Text
                    if viewModel.isSessionRunning && !viewModel.feedback.isEmpty {
                        Text(viewModel.feedback)
                            .font(.system(size: 42, weight: .heavy, design: .rounded))
                            .foregroundColor(.white)
                            .multilineTextAlignment(.center)
                            .lineLimit(3)
                            .minimumScaleFactor(0.5)
                            .shadow(color: .black.opacity(0.8), radius: 8, x: 0, y: 4)
                            .padding(.horizontal, 24)
                            .animation(.spring(), value: viewModel.feedback)
                    }

                    Spacer()

                    bottomPanel
                        .padding(.horizontal, 16)
                        .padding(.bottom, 16)
                }

                if !viewModel.isSessionRunning {
                    guideCenterOverlay
                        .transition(.opacity)
                }

                if viewModel.showSquatPreview {
                    exercisePreviewOverlay(
                        title: "Next: Squat\nStarting in \(viewModel.squatPreviewSeconds)s",
                        guideContent: ExerciseGuideOverlay(
                            frames: ["squat_01", "squat_02", "squat_03"],
                            label: "Lower -> Hold -> Stand up"
                        )
                    )
                    .transition(.opacity)
                }

                if viewModel.showPlankPreview {
                    exercisePreviewOverlay(
                        title: "Next: Plank\nStarting in \(viewModel.plankPreviewSeconds)s",
                        guideContent: ExerciseGuideOverlay(
                            frames: ["plank_01"],
                            label: "Hold straight"
                        )
                    )
                    .transition(.opacity)
                }
            }
            .navigationBarBackButtonHidden(true)
            .navigationDestination(isPresented: $viewModel.navigateToResult) {
                WorkoutResultView(summary: viewModel.sessionSummary) {
                    dismiss()
                }
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

    // MARK: - Center Guide (shown before session starts)

    private var guideCenterOverlay: some View {
        VStack(spacing: 12) {
            currentGuideOverlay

                Text("Set camera to SIDE VIEW\nPress Start when ready")
                    .font(.subheadline.weight(.bold))
                .foregroundColor(.white)
                .multilineTextAlignment(.center)
                .padding(.horizontal, 20)
                .padding(.vertical, 14)
                .background(.ultraThinMaterial)
                .environment(\.colorScheme, .dark)
                .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                .shadow(color: .black.opacity(0.5), radius: 8, x: 0, y: 4)
        }
        .padding(.horizontal, 18)
        .offset(y: -40)
    }

    // MARK: - Exercise Preview Overlay

    /// Generic preview overlay shown during exercise transitions (countdown + guide).
    private func exercisePreviewOverlay<Content: View>(
        title: String,
        guideContent: Content
    ) -> some View {
        ZStack {
            Color.black.opacity(0.45).ignoresSafeArea()

            VStack(spacing: 12) {
                guideContent

                Text(title)
                    .font(.headline.weight(.bold))
                    .foregroundColor(.white)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 20)
                    .padding(.vertical, 14)
                    .background(.ultraThinMaterial)
                    .environment(\.colorScheme, .dark)
                    .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                    .shadow(color: .black.opacity(0.5), radius: 8, x: 0, y: 4)
            }
            .padding(.horizontal, 18)
            .offset(y: -40)
        }
    }

    // MARK: - Bottom Panel

    private var bottomPanel: some View {
        VStack(spacing: 12) {
            HStack(spacing: 12) {
                StatChip(title: "Total", value: viewModel.totalReps, tint: .white)
                StatChip(title: "Correct", value: viewModel.correctReps, tint: .green)
                StatChip(title: "Incorrect", value: viewModel.incorrectReps, tint: .red)
            }

            progressCard

            Button {
                if viewModel.isSessionRunning {
                    viewModel.finishSession()
                } else {
                    viewModel.startSession()
                }
            } label: {
                Text(viewModel.isSessionRunning ? "End Workout" : "Start Workout")
                    .font(.headline.weight(.bold))
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 16)
                    .foregroundColor(viewModel.isSessionRunning ? .white : .black)
                    .background(viewModel.isSessionRunning ? Color.red.gradient : Color.yellow.gradient)
                    .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
                    .shadow(color: (viewModel.isSessionRunning ? Color.red : Color.yellow).opacity(0.3), radius: 10, x: 0, y: 5)
            }
        }
    }

    // MARK: - Progress Cards

    /// Selects the appropriate progress card for the current exercise mode.
    private var progressCard: some View {
        Group {
            switch viewModel.mode {
            case .wallSit:
                ExerciseProgressCard(
                    title: wallSitTitleText,
                    detail: wallSitDetailText,
                    progress: viewModel.wallSitProgress01
                )
            case .squat:
                ExerciseProgressCard(
                    title: squatTitleText,
                    detail: "\(viewModel.correctReps) / \(viewModel.squatTargetCorrectReps) correct",
                    progress: viewModel.squatProgress01
                )
            case .plank:
                ExerciseProgressCard(
                    title: plankTitleText,
                    detail: String(format: "%.1f", viewModel.plankCorrectSeconds) + " / " + String(format: "%.1f", viewModel.plankTargetSeconds) + " s",
                    progress: viewModel.plankProgress01
                )
            case .lunges:
                ExerciseProgressCard(
                    title: "Lunges Goal",
                    detail: "\(viewModel.correctReps) / \(viewModel.lungesTargetCorrectReps) correct",
                    progress: viewModel.lungesProgress01
                )
            }
        }
    }

    // MARK: - Progress Card Title/Detail Helpers

    private var wallSitTitleText: String {
        if viewModel.passedWallSit { return "PASSED (Switch to Squat)" }
        if viewModel.wallSitCountingActive { return "Hold Correct (5s)" }
        return "Get In Position"
    }

    private var wallSitDetailText: String {
        if viewModel.wallSitCountingActive {
            return "\(String(format: "%.1f", viewModel.correctSeconds)) / \(String(format: "%.1f", viewModel.targetSeconds)) s"
        }
        return "\(viewModel.wallSitConsecutiveCorrect)"
    }

    private var squatTitleText: String {
        if viewModel.correctReps >= viewModel.squatTargetCorrectReps {
            return "PASSED (Switch to Plank)"
        }
        return "Squat Goal"
    }

    private var plankTitleText: String {
        if viewModel.passedPlank { return "PASSED" }
        return "Plank Goal"
    }

    // MARK: - Guide Overlay Selection

    @ViewBuilder
    private var currentGuideOverlay: some View {
        switch viewModel.mode {
        case .wallSit:
            ExerciseGuideOverlay(
                frames: ["wall_01", "wall_02", "wall_03"],
                label: "Stand -> Lower -> Hold"
            )
        case .squat:
            ExerciseGuideOverlay(
                frames: ["squat_01", "squat_02", "squat_03"],
                label: "Lower -> Hold -> Stand up"
            )
        case .plank:
            ExerciseGuideOverlay(
                frames: ["plank_01"],
                label: "Hold straight"
            )
        case .lunges:
            ExerciseGuideOverlay(
                frames: ["squat_01", "squat_02"],
                label: "Step forward -> Lower -> Push back"
            )
        }
    }
}

// MARK: - Top HUD

/// Status bar at the top showing the set title.
private struct TopHUD: View {
    let isSessionRunning: Bool
    let setTitle: String

    var body: some View {
        HStack {
            Spacer()
            Text(setTitle)
                .font(.subheadline.weight(.semibold))
                .foregroundColor(.white)
                .lineLimit(1)
                .padding(.horizontal, 16)
                .padding(.vertical, 8)
                .background(.ultraThinMaterial)
                .environment(\.colorScheme, .dark)
                .clipShape(Capsule())
        }
    }
}

// MARK: - Exercise Guide Overlay

/// Animated exercise guide that cycles through demonstration frames with a text label.
/// Used both in the pre-session overlay and in exercise transition previews.
///
/// For single-frame exercises (e.g., plank), displays a static image without animation.
private struct ExerciseGuideOverlay: View {
    let frames: [String]
    let label: String

    var body: some View {
        if frames.count > 1 {
            animatedContent
        } else {
            staticContent
        }
    }

    private var animatedContent: some View {
        TimelineView(.periodic(from: Date(), by: 1.1)) { context in
            let idx = Int(context.date.timeIntervalSince1970 / 1.1) % frames.count
            VStack(spacing: 10) {
                Image(frames[idx])
                    .resizable()
                    .scaledToFit()
                    .frame(maxWidth: 260, maxHeight: 200)
                    .shadow(radius: 10)
                    .animation(.easeInOut(duration: 0.2), value: idx)

                guideLabel
            }
        }
    }

    private var staticContent: some View {
        VStack(spacing: 10) {
            Image(frames.first ?? "")
                .resizable()
                .scaledToFit()
                .frame(maxWidth: 260, maxHeight: 200)
                .shadow(radius: 10)

            guideLabel
        }
    }

    private var guideLabel: some View {
        Text(label)
            .font(.caption.weight(.semibold))
            .foregroundColor(.white)
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(Color.black.opacity(0.45))
            .clipShape(Capsule())
    }
}

// MARK: - Stat Chip

/// Small rounded chip displaying a labeled numeric value (e.g., "Total: 5").
private struct StatChip: View {
    let title: String
    let value: Int
    let tint: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.caption.weight(.medium))
                .foregroundColor(.white.opacity(0.7))
            Text("\(value)")
                .font(.system(size: 26, weight: .bold, design: .rounded))
                .monospacedDigit()
                .foregroundColor(tint)
        }
        .padding(.vertical, 12)
        .padding(.horizontal, 14)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.ultraThinMaterial)
        .environment(\.colorScheme, .dark)
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
    }
}

// MARK: - Exercise Progress Card

/// Card showing exercise title, detail text, and a progress bar.
/// Shared across all exercise modes (wall-sit, squat, plank, lunges).
private struct ExerciseProgressCard: View {
    let title: String
    let detail: String
    let progress: Double

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(title)
                    .font(.subheadline.weight(.bold))
                    .foregroundColor(.white)
                Spacer()
                Text(detail)
                    .font(.subheadline.weight(.bold))
                    .foregroundColor(.white.opacity(0.95))
                    .monospacedDigit()
            }

            ProgressBarView(progress: progress, height: 12, fillColor: progress >= 1.0 ? .green : .yellow)
        }
        .padding(.vertical, 14)
        .padding(.horizontal, 16)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.ultraThinMaterial)
        .environment(\.colorScheme, .dark)
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
    }
}

// MARK: - Shared Progress Bar

/// Reusable animated progress bar used across session, result, and history views.
struct ProgressBarView: View {
    let progress: Double
    var height: CGFloat = 10
    var fillColor: Color = .yellow
    var trackColor: Color = .white.opacity(0.25)

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                RoundedRectangle(cornerRadius: height)
                    .fill(trackColor)

                RoundedRectangle(cornerRadius: height)
                    .fill(fillColor)
                    .frame(width: max(0, geo.size.width * CGFloat(min(max(0, progress), 1.0))))
                    .animation(.easeInOut(duration: 0.25), value: progress)
            }
        }
        .frame(height: height)
    }
}
