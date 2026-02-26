import SwiftUI
import Charts

struct ProgressView: View {
    @StateObject private var viewModel = ProgressViewModel()

    var body: some View {
        ScrollView {
                VStack(alignment: .leading, spacing: 24) {
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

                    Text("Workout Summary")
                        .font(.headline)
                        .foregroundColor(.white)

                    if viewModel.workoutSummaries.isEmpty && !viewModel.isLoading {
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
                            Text("Loading…")
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
                            Text("Make sure you’re signed in, then complete a workout and tap End session. It will appear here.")
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

    /// ดึง URL จากข้อความ error (เช่น Firestore index link) เพื่อให้กดเปิดได้
    private func urlFromError(_ err: String) -> URL? {
        guard let start = err.range(of: "https://")?.lowerBound else { return nil }
        let rest = String(err[start...])
        let end = rest.firstIndex(where: { $0.isWhitespace || $0 == ")" }) ?? rest.endIndex
        return URL(string: String(rest[..<end]))
    }
}

// MARK: - History list row
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

// MARK: - Detail: แสดงแบบเดียวกับ Workout Summary (เวลา, แคล, รายการท่า)
struct HistoryDetailView: View {
    let item: WorkoutHistoryItem

    var body: some View {
        ScrollView(showsIndicators: false) {
            VStack(alignment: .leading, spacing: 24) {
                // หัว: เวลา + แคล (แบบ Workout Result)
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
                        HistoryExerciseCard(exercise: ex)
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
    }
}

// MARK: - การ์ดหนึ่งท่า (รูปแบบเดียวกับ Workout Result)
private struct HistoryExerciseCard: View {
    let exercise: ExerciseRecord

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 12) {
                Image(systemName: iconName)
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
                    progressBar(progress: target > 0 ? Double(correct) / Double(target) : 0)
                    Text("\(correct) / \(target) reps")
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .foregroundColor(.white)
                }
                if let total = exercise.totalReps, total > 0, let inc = exercise.incorrectReps {
                    Text("Total \(total) reps · \(inc) incorrect")
                        .font(.caption)
                        .foregroundColor(.gray)
                }
            } else {
                if let dur = exercise.durationSeconds, let tgt = exercise.targetSeconds {
                    progressBar(progress: tgt > 0 ? dur / tgt : 0)
                    Text("\(Int(dur)) sec / \(Int(tgt)) sec")
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .foregroundColor(.white)
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
                        Text("• \(e.reason) — \(e.count) times")
                            .font(.subheadline)
                            .foregroundColor(.orange.opacity(0.95))
                    }
                }
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("Mistake timeline")
                    .font(.caption)
                    .fontWeight(.medium)
                    .foregroundColor(.gray)
                if exercise.mistakes.isEmpty {
                    Text("No mistake timestamps")
                        .font(.subheadline)
                        .foregroundColor(.gray.opacity(0.7))
                } else {
                    ForEach(Array(exercise.mistakes.enumerated()), id: \.offset) { _, event in
                        Text(mistakeTimelineText(event))
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

    private var iconName: String {
        switch exercise.name.lowercased() {
        case "wall-sit": return "figure.stand"
        case "squat": return "figure.strengthtraining.traditional"
        default: return "figure.strengthtraining.traditional"
        }
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

    private func formatTimelineSecond(_ second: Int) -> String {
        let m = max(0, second) / 60
        let s = max(0, second) % 60
        return String(format: "%02d:%02d", m, s)
    }

    private func mistakeTimelineText(_ event: MistakeEventRecord) -> String {
        if let rep = event.repNumber {
            return "• Rep \(rep) — \(event.reason) (\(formatTimelineSecond(event.atSecond)))"
        }
        return "• \(formatTimelineSecond(event.atSecond)) — \(event.reason)"
    }
}
