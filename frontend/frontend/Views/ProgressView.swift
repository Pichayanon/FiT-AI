import SwiftUI
import Charts

struct ProgressView: View {
    @StateObject private var viewModel = ProgressViewModel()

    var body: some View {
        NavigationView {
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

                    Spacer()
                }
                .padding()
            }
            .background(Color.black.edgesIgnoringSafeArea(.all))
        }
    }
}
