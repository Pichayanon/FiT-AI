import SwiftUI
import Charts

struct ProgressView: View {
    let calorieStats: [DailyCalorie] = [
        .init(date: Date().addingTimeInterval(-6*86400), calories: 120),
        .init(date: Date().addingTimeInterval(-5*86400), calories: 150),
        .init(date: Date().addingTimeInterval(-4*86400), calories: 200),
        .init(date: Date().addingTimeInterval(-3*86400), calories: 180),
        .init(date: Date().addingTimeInterval(-2*86400), calories: 160),
        .init(date: Date().addingTimeInterval(-1*86400), calories: 210),
        .init(date: Date(), calories: 190),
    ]
    
    let workoutSummaries: [WorkoutSetSummary] = [
        .init(name: "Beginner Level 1", timesCompleted: 3, totalCalories: 450),
        .init(name: "Beginner Level 2", timesCompleted: 2, totalCalories: 320),
        .init(name: "Intermediate Level 1", timesCompleted: 1, totalCalories: 200)
    ]

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(alignment: .leading, spacing: 24) {
                    
                    Text("Calories Burned This Week")
                        .font(.headline)
                        .foregroundColor(.white)

                    Chart {
                        ForEach(calorieStats) { day in
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

                    ForEach(workoutSummaries) { summary in
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

struct DailyCalorie: Identifiable {
    let id = UUID()
    let date: Date
    let calories: Int
}

struct WorkoutSetSummary: Identifiable {
    let id = UUID()
    let name: String
    let timesCompleted: Int
    let totalCalories: Int
}
