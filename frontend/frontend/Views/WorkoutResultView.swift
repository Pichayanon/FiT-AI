import SwiftUI

/// Displays the post-workout result screen with exercise summaries, stats, and error details.
struct WorkoutResultView: View {
    @StateObject private var viewModel: WorkoutResultViewModel
    @Environment(\.dismiss) private var dismiss
    private let onBackToHome: (() -> Void)?

    init(summary: SessionSummary, onBackToHome: (() -> Void)? = nil) {
        _viewModel = StateObject(wrappedValue: WorkoutResultViewModel(summary: summary))
        self.onBackToHome = onBackToHome
    }

    var body: some View {
        VStack(spacing: 0) {
            ScrollView(showsIndicators: false) {
                VStack(alignment: .leading, spacing: 24) {
                    // Completion header with primary stats
                    VStack(spacing: 16) {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.system(size: 56))
                            .foregroundStyle(.yellow)

                        Text("Workout complete!")
                            .font(.title2)
                            .fontWeight(.bold)
                            .foregroundColor(.white)

                        HStack(spacing: 16) {
                            StatPill(icon: "clock.fill", value: viewModel.formattedTotalTime)
                            StatPill(icon: "flame.fill", value: "\(viewModel.summary.estimatedCalories) kcal")
                        }
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.top, 12)
                    .padding(.bottom, 8)

                    Text("Summary")
                        .font(.headline)
                        .foregroundColor(.gray)
                        .padding(.horizontal, 4)

                    LazyVStack(spacing: 12) {
                        ForEach(viewModel.summary.items) { item in
                            SummaryExerciseCard(item: item)
                        }
                    }
                    .padding(.bottom, 32)
                }
                .padding(.horizontal, 20)
            }

            // Sticky primary CTA
            Button(action: {
                if let onBackToHome {
                    onBackToHome()
                } else {
                    dismiss()
                }
            }) {
                Text("Back to Home")
                    .font(.headline)
                    .frame(maxWidth: .infinity)
                    .frame(height: 52)
                    .background(Color.yellow)
                    .foregroundColor(.black)
                    .cornerRadius(14)
            }
            .buttonStyle(.plain)
            .padding(.horizontal, 20)
            .padding(.top, 12)
            .padding(.bottom, 28)
            .background(Color.black)
        }
        .background(Color.black.ignoresSafeArea())
    }
}

// MARK: - Stat Pill

/// Compact pill showing an SF Symbol icon with a text value (e.g., clock + "2 min 30 sec").
fileprivate struct StatPill: View {
    let icon: String
    let value: String

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: icon)
                .font(.subheadline)
                .foregroundColor(.yellow)
            Text(value)
                .font(.subheadline)
                .fontWeight(.medium)
                .foregroundColor(.white)
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
        .background(Color.white.opacity(0.08))
        .cornerRadius(12)
    }
}

// MARK: - Summary Exercise Card

/// Card displaying detailed stats for one exercise (reps/hold, errors, mistake timeline).
fileprivate struct SummaryExerciseCard: View {
    let item: ExerciseSummaryItem

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 12) {
                Image(systemName: "figure.strengthtraining.traditional")
                    .font(.title2)
                    .foregroundColor(.yellow)
                    .frame(width: 44, height: 44)
                    .background(Color.white.opacity(0.08))
                    .clipShape(Circle())

                Text(item.displayName)
                    .font(.title3)
                    .fontWeight(.semibold)
                    .foregroundColor(.white)
                Spacer()
            }

            switch item {
            case .movement(_, let totalReps, let correctReps, let incorrectReps, let targetCorrectReps, let errors, let mistakes):
                VStack(alignment: .leading, spacing: 8) {
                    ProgressBarView(
                        progress: targetCorrectReps > 0 ? Double(correctReps) / Double(targetCorrectReps) : 0,
                        height: 6,
                        fillColor: .yellow,
                        trackColor: .white.opacity(0.12)
                    )
                    Text("\(correctReps) / \(targetCorrectReps) reps")
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .foregroundColor(.white)
                    if totalReps > 0 {
                        Text("Total \(totalReps) reps - \(incorrectReps) incorrect")
                            .font(.caption)
                            .foregroundColor(.gray)
                    }
                    ErrorsBlock(errors: errors)
                    MistakesBlock(mistakes: mistakes)
                }

            case .isometric(_, let durationSeconds, let targetSeconds, let errors, _):
                VStack(alignment: .leading, spacing: 8) {
                    ProgressBarView(
                        progress: targetSeconds > 0 ? durationSeconds / targetSeconds : 0,
                        height: 6,
                        fillColor: .yellow,
                        trackColor: .white.opacity(0.12)
                    )
                    Text("\(formatSeconds(durationSeconds)) / \(formatSeconds(targetSeconds))")
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .foregroundColor(.white)
                    ErrorsBlock(errors: errors)
                }
            }
        }
        .padding(16)
        .background(Color.white.opacity(0.06))
        .cornerRadius(16)
    }

    private func formatSeconds(_ s: Double) -> String {
        "\(Int(round(s))) sec"
    }
}

// MARK: - Errors Block

/// Lists error reasons with their counts, or "None" if empty.
fileprivate struct ErrorsBlock: View {
    let errors: [ErrorCount]

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("What went wrong")
                .font(.caption)
                .fontWeight(.medium)
                .foregroundColor(.gray)
            if errors.isEmpty {
                Text("None")
                    .font(.subheadline)
                    .foregroundColor(.gray.opacity(0.7))
            } else {
                ForEach(errors) { e in
                    Text("\u{2022} \(e.reason) -- \(e.count) times")
                        .font(.subheadline)
                        .foregroundColor(.orange.opacity(0.95))
                }
            }
        }
    }
}

// MARK: - Mistakes Block

/// Lists timestamped mistake events, or a "no mistakes" message if empty.
fileprivate struct MistakesBlock: View {
    let mistakes: [MistakeEvent]

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Mistake timeline")
                .font(.caption)
                .fontWeight(.medium)
                .foregroundColor(.gray)
            if mistakes.isEmpty {
                Text("No mistake timestamps")
                    .font(.subheadline)
                    .foregroundColor(.gray.opacity(0.7))
            } else {
                ForEach(mistakes) { event in
                    Text(mistakeTimelineText(event))
                        .font(.subheadline)
                        .foregroundColor(.orange.opacity(0.95))
                }
            }
        }
    }

    private func mistakeTimelineText(_ event: MistakeEvent) -> String {
        if let rep = event.repNumber {
            return "\u{2022} \(event.reason) -- Rep \(rep)"
        }
        return "\u{2022} \(event.reason)"
    }
}
