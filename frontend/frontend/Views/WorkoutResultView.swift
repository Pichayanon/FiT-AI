import SwiftUI

struct WorkoutResultView: View {
    @StateObject private var viewModel: WorkoutResultViewModel
    @Environment(\.dismiss) private var dismiss

    init(summary: SessionSummary) {
        _viewModel = StateObject(wrappedValue: WorkoutResultViewModel(summary: summary))
    }

    var body: some View {
        VStack(spacing: 0) {
            ScrollView(showsIndicators: false) {
                VStack(alignment: .leading, spacing: 24) {
                    // Success moment: completion + primary stats
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
            Button(action: { dismiss() }) {
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

// MARK: - Top stat pill (time / calories)
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

// MARK: - One exercise card (stats + progress + what went wrong)
fileprivate struct SummaryExerciseCard: View {
    let item: ExerciseSummaryItem

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 12) {
                Image(systemName: iconName)
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
            case .movement(_, let totalReps, let correctReps, let incorrectReps, let targetCorrectReps, let errors):
                VStack(alignment: .leading, spacing: 8) {
                    progressBar(progress: targetCorrectReps > 0 ? Double(correctReps) / Double(targetCorrectReps) : 0)
                    Text("\(correctReps) / \(targetCorrectReps) reps")
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .foregroundColor(.white)
                    if totalReps > 0 {
                        Text("Total \(totalReps) reps · \(incorrectReps) incorrect")
                            .font(.caption)
                            .foregroundColor(.gray)
                    }
                    errorsBlock(errors)
                }

            case .isometric(_, let durationSeconds, let targetSeconds, let errors):
                VStack(alignment: .leading, spacing: 8) {
                    progressBar(progress: targetSeconds > 0 ? durationSeconds / targetSeconds : 0)
                    Text("\(formatSeconds(durationSeconds)) / \(formatSeconds(targetSeconds))")
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .foregroundColor(.white)
                    errorsBlock(errors)
                }
            }
        }
        .padding(16)
        .background(Color.white.opacity(0.06))
        .cornerRadius(16)
    }

    private func progressBar(progress: Double) -> some View {
        let p = min(1, max(0, progress))
        return GeometryReader { geo in
            ZStack(alignment: .leading) {
                RoundedRectangle(cornerRadius: 4)
                    .fill(Color.white.opacity(0.12))
                RoundedRectangle(cornerRadius: 4)
                    .fill(Color.yellow)
                    .frame(width: geo.size.width * p)
            }
        }
        .frame(height: 6)
        .frame(maxWidth: .infinity)
    }

    /// What went wrong — framed as feedback, not blame
    private func errorsBlock(_ errors: [ErrorCount]) -> some View {
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
                    Text("• \(e.reason) — \(e.count) times")
                        .font(.subheadline)
                        .foregroundColor(.orange.opacity(0.95))
                }
            }
        }
    }

    private var iconName: String {
        switch item.displayName.lowercased() {
        case "wall-sit": return "figure.strengthtraining.traditional"
        case "squat": return "figure.strengthtraining.traditional"
        default: return "fxklellellliizzzzzdddigure.strengthtraining.traditional"
        }
    }

    private func formatSeconds(_ s: Double) -> String {
        let n = Int(round(s))
        return "\(n) sec"
    }
}
