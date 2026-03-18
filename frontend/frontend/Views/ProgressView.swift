import SwiftUI
import Charts

/// Shows workout progress: weekly calorie chart, workout summaries, and session history.
struct ProgressView: View {
    @StateObject private var viewModel = ProgressViewModel()

    var body: some View {
        ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    weeklyChartSection
                    summarySection
                    historySection
                    Spacer()
                }
                .padding()
            }
            .background(Color.black.edgesIgnoringSafeArea(.all))
            .navigationTitle("Progress")
            .navigationBarTitleDisplayMode(.inline)
            .refreshable { await viewModel.load() }
            .task { await viewModel.load() }
    }
    
    // MARK: - Weekly Chart Section
    
    private var weeklyChartSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Calories Burned This Week")
                .font(.headline)
                .foregroundColor(.white)

            Chart {
                ForEach(viewModel.calorieStats) { day in
                    BarMark(
                        x: .value("Day", day.date, unit: .day),
                        y: .value("Calories", day.calories)
                    )
                    .foregroundStyle(.orange)
                    .annotation(position: .overlay, alignment: .center) {
                        NavigationLink(destination: DailyDetailView(date: day.date)) {
                            Rectangle()
                                .foregroundColor(.clear)
                                .frame(width: 30, height: 200)
                        }
                        .buttonStyle(.plain)
                    }
                    .annotation(position: .top) {
                        Text("\(day.calories)")
                            .font(.caption)
                            .foregroundColor(.white)
                    }
                }
            }
            .chartYScale(domain: 0...300)
            .chartXAxis {
                AxisMarks(values: .stride(by: .day)) { value in
                    AxisGridLine()
                    AxisValueLabel(format: .dateTime.weekday(.narrow))
                }
            }
            .frame(height: 220)
            .background(Color.gray.opacity(0.15))
            .cornerRadius(12)
        }
    }
    
    // MARK: - Summary Section
    
    private var summarySection: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Workout Summary")
                .font(.headline)
                .foregroundColor(.white)

            if viewModel.isLoading && viewModel.workoutSummaries.isEmpty {
                HStack(spacing: 8) {
                    SwiftUI.ProgressView()
                        .progressViewStyle(CircularProgressViewStyle(tint: .yellow))
                    Text("Loading...")
                        .font(.subheadline)
                        .foregroundColor(.gray)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 24)
            } else if viewModel.workoutSummaries.isEmpty {
                Text("No workouts yet")
                    .font(.subheadline)
                    .foregroundColor(.gray)
                    .padding(.vertical, 8)
            } else {
                ForEach(viewModel.workoutSummaries) { summary in
                    HStack {
                        VStack(alignment: .leading) {
                            Text(summary.name)
                                .font(.subheadline)
                                .foregroundColor(.yellow)
                            Text("Completed \(summary.timesCompleted) times")
                                .font(.caption)
                                .foregroundColor(.gray)
                        }
                        Spacer()
                        Text("\(summary.totalCalories) kcal")
                            .font(.body)
                            .foregroundColor(.orange)
                    }
                    .padding()
                    .background(Color.gray.opacity(0.15))
                    .cornerRadius(12)
                }
            }
        }
    }
    
    // MARK: - History Section
    
    private var historySection: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("History")
                .font(.headline)
                .foregroundColor(.white)

            if let err = viewModel.errorMessage {
                VStack(alignment: .leading, spacing: 10) {
                    Text("Could not load history")
                        .font(.subheadline)
                        .foregroundColor(.orange)
                    Text(err)
                        .font(.caption)
                        .foregroundColor(.gray)
                    HStack(spacing: 12) {
                        if let url = urlFromError(err) {
                            Button("Open link to create index") {
                                UIApplication.shared.open(url)
                            }
                            .font(.subheadline)
                            .foregroundColor(.yellow)
                        }
                        Button("Retry") { Task { await viewModel.load() } }
                            .font(.subheadline)
                            .foregroundColor(.yellow)
                    }
                }
                .padding()
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.orange.opacity(0.1))
                .cornerRadius(12)
            } else if viewModel.isLoading && viewModel.historyItems.isEmpty {
                HStack(spacing: 8) {
                    SwiftUI.ProgressView()
                        .progressViewStyle(CircularProgressViewStyle(tint: .yellow))
                    Text("Loading...")
                        .font(.subheadline)
                        .foregroundColor(.gray)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 24)
            } else if viewModel.historyItems.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    Text("No history yet")
                        .font(.subheadline)
                        .foregroundColor(.gray)
                    Text("Make sure you're signed in, then complete a workout and tap End Workout. It will appear here.")
                        .font(.caption)
                        .foregroundColor(.gray.opacity(0.9))
                }
                .padding(.vertical, 16)
            } else {
                VStack(spacing: 10) {
                    ForEach(viewModel.historyItems) { item in
                        NavigationLink(destination: HistoryDetailView(item: item)) {
                            HistoryRow(item: item)
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
        }
    }
    
    /// Extracts a URL from an error message (e.g., Firestore index creation link).
    private func urlFromError(_ err: String) -> URL? {
        guard let start = err.range(of: "https://")?.lowerBound else { return nil }
        let rest = String(err[start...])
        let end = rest.firstIndex(where: { $0.isWhitespace || $0 == ")" }) ?? rest.endIndex
        return URL(string: String(rest[..<end]))
    }
}

// MARK: - History Row

/// One row in the workout history list showing set title, date, calories, and duration.
private struct HistoryRow: View {
    let item: WorkoutHistoryItem

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: "checkmark.circle.fill")
                .font(.title2)
                .foregroundColor(.yellow)
            VStack(alignment: .leading, spacing: 4) {
                Text(item.setTitle)
                    .font(.subheadline)
                    .fontWeight(.semibold)
                    .foregroundColor(.white)
                Text(item.relativeDate)
                    .font(.caption)
                    .foregroundColor(.gray)
            }
            Spacer()
            VStack(alignment: .trailing, spacing: 2) {
                Text("\(item.estimatedCalories) kcal")
                    .font(.subheadline)
                    .foregroundColor(.orange)
                Text(item.formattedDuration)
                    .font(.caption)
                    .foregroundColor(.gray)
            }
            Image(systemName: "chevron.right")
                .font(.caption)
                .foregroundColor(.gray)
        }
        .padding()
        .background(Color.gray.opacity(0.15))
        .cornerRadius(12)
    }
}

// MARK: - History Detail View

/// Detailed view for a past workout session, showing per-exercise stats and errors.
struct HistoryDetailView: View {
    let item: WorkoutHistoryItem
    @State private var selectedPlayback: SessionMistakePlayback?

    var body: some View {
        ScrollView(showsIndicators: false) {
            VStack(alignment: .leading, spacing: 24) {
                // Header: time + calories
                VStack(spacing: 12) {
                    HStack(spacing: 16) {
                        HStack(spacing: 8) {
                            Image(systemName: "clock.fill")
                                .foregroundColor(.yellow)
                            Text("\(item.totalTimeSeconds / 60) min \(item.totalTimeSeconds % 60) s")
                                .font(.subheadline)
                                .fontWeight(.medium)
                                .foregroundColor(.white)
                        }
                        .padding(.horizontal, 16)
                        .padding(.vertical, 10)
                        .background(Color.white.opacity(0.08))
                        .cornerRadius(12)
                        HStack(spacing: 8) {
                            Image(systemName: "flame.fill")
                                .foregroundColor(.yellow)
                            Text("\(item.estimatedCalories) kcal")
                                .font(.subheadline)
                                .fontWeight(.medium)
                                .foregroundColor(.white)
                        }
                        .padding(.horizontal, 16)
                        .padding(.vertical, 10)
                        .background(Color.white.opacity(0.08))
                        .cornerRadius(12)
                    }
                    Text(item.formattedDate)
                        .font(.caption)
                        .foregroundColor(.gray)
                }
                .frame(maxWidth: .infinity)
                .padding(.top, 8)

                Text("Summary")
                    .font(.headline)
                    .foregroundColor(.gray)

                LazyVStack(spacing: 12) {
                    ForEach(Array(item.record.exercises.enumerated()), id: \.offset) { _, ex in
                        HistoryExerciseCard(
                            exercise: ex,
                            videoURL: SessionVideoStore.recordingURL(for: item.record.sessionVideoFileName),
                            selectedPlayback: $selectedPlayback
                        )
                    }
                }
                .padding(.bottom, 32)
            }
            .padding(.horizontal, 20)
        }
        .background(Color.black.ignoresSafeArea())
        .navigationTitle(item.setTitle)
        .navigationBarTitleDisplayMode(.inline)
        .foregroundColor(.white)
        .sheet(item: $selectedPlayback) { playback in
            SessionVideoPlaybackView(playback: playback)
        }
    }
}

// MARK: - History Exercise Card

/// Card showing detailed stats for one exercise in a past session.
private struct HistoryExerciseCard: View {
    let exercise: ExerciseRecord
    let videoURL: URL?
    @Binding var selectedPlayback: SessionMistakePlayback?

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 12) {
                Image(systemName: "figure.strengthtraining.traditional")
                    .font(.title2)
                    .foregroundColor(.yellow)
                    .frame(width: 44, height: 44)
                    .background(Color.white.opacity(0.08))
                    .clipShape(Circle())
                Text(exercise.name)
                    .font(.title3)
                    .fontWeight(.semibold)
                    .foregroundColor(.white)
                Spacer()
            }

            if exercise.type == "movement" {
                if let correct = exercise.correctReps, let target = exercise.targetCorrectReps {
                    ProgressBarView(
                        progress: target > 0 ? Double(correct) / Double(target) : 0,
                        height: 6,
                        fillColor: .yellow,
                        trackColor: .white.opacity(0.12)
                    )
                    Text("\(correct) / \(target) reps")
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .foregroundColor(.white)
                }
                if let total = exercise.totalReps, total > 0, let inc = exercise.incorrectReps {
                    Text("Total \(total) reps - \(inc) incorrect")
                        .font(.caption)
                        .foregroundColor(.gray)
                }
            } else {
                if let dur = exercise.durationSeconds, let tgt = exercise.targetSeconds {
                    ProgressBarView(
                        progress: tgt > 0 ? dur / tgt : 0,
                        height: 6,
                        fillColor: .yellow,
                        trackColor: .white.opacity(0.12)
                    )
                    Text("\(Int(dur)) sec / \(Int(tgt)) sec")
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .foregroundColor(.white)
                }
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("Mistake timeline")
                    .font(.caption)
                    .fontWeight(.medium)
                    .foregroundColor(.gray)
                if exercise.mistakes.isEmpty {
                    Text("No mistake records")
                        .font(.subheadline)
                        .foregroundColor(.gray.opacity(0.7))
                } else {
                    ForEach(Array(exercise.mistakes.enumerated()), id: \.offset) { _, event in
                        if let videoURL {
                            Button {
                                selectedPlayback = makePlayback(for: event, videoURL: videoURL)
                            } label: {
                                HStack(spacing: 10) {
                                    VStack(alignment: .leading, spacing: 2) {
                                        Text(formatMistakeTimeline(event))
                                            .font(.subheadline)
                                            .foregroundColor(.orange.opacity(0.95))
                                        Text("Tap to replay this mistake")
                                            .font(.caption)
                                            .foregroundColor(.gray)
                                    }
                                    Spacer()
                                    Image(systemName: "play.circle.fill")
                                        .font(.title3)
                                        .foregroundColor(.yellow)
                                }
                                .padding(.vertical, 8)
                                .padding(.horizontal, 10)
                                .background(Color.white.opacity(0.04))
                                .cornerRadius(12)
                            }
                            .buttonStyle(.plain)
                        } else {
                            Text(formatMistakeTimeline(event))
                                .font(.subheadline)
                                .foregroundColor(.orange.opacity(0.95))
                        }
                    }

                    if videoURL == nil {
                        Text("Video unavailable for this session")
                            .font(.caption)
                            .foregroundColor(.gray)
                    }
                }
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("What went wrong")
                    .font(.caption)
                    .fontWeight(.medium)
                    .foregroundColor(.gray)
                if exercise.errors.isEmpty {
                    Text("None")
                        .font(.subheadline)
                        .foregroundColor(.gray.opacity(0.7))
                } else {
                    ForEach(Array(exercise.errors.enumerated()), id: \.offset) { _, e in
                        Text("\u{2022} \(e.reason) — \(e.count) times")
                            .font(.subheadline)
                            .foregroundColor(.orange.opacity(0.95))
                    }
                }
            }
        }
        .padding(16)
        .background(Color.white.opacity(0.06))
        .cornerRadius(16)
    }

    private func formatMistakeTimeline(_ event: MistakeEventRecord) -> String {
        if let rep = event.repNumber {
            return "\u{2022} \(event.reason) — Rep \(rep)"
        }
        return "\u{2022} \(event.reason)"
    }

    private func makePlayback(for event: MistakeEventRecord, videoURL: URL) -> SessionMistakePlayback {
        let subtitle: String
        if let rep = event.repNumber {
            subtitle = "\(exercise.name) • Rep \(rep)"
        } else {
            subtitle = exercise.name
        }

        return SessionMistakePlayback(
            title: event.reason,
            subtitle: subtitle,
            videoURL: videoURL,
            atSecond: event.atSecond
        )
    }
}
