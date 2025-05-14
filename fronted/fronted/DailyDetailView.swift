import SwiftUI

struct DailyDetailView: View {
    let date: Date

    let workoutSets: [WorkoutSetSummary] = [
        .init(name: "Beginner Level 1", timesCompleted: 1, totalCalories: 150),
        .init(name: "Beginner Level 2", timesCompleted: 1, totalCalories: 120)
    ]

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            Text(date.formatted(.dateTime.weekday().month().day()))
                .font(.title)
                .foregroundColor(.white)

            Text("Total Burn: \(workoutSets.map { $0.totalCalories }.reduce(0, +)) kcal")
                .font(.headline)
                .foregroundColor(.orange)

            Divider().background(Color.gray)

            ForEach(workoutSets) { set in
                VStack(alignment: .leading, spacing: 4) {
                    Text(set.name)
                        .font(.headline)
                        .foregroundColor(.yellow)
                    Text("Completed \(set.timesCompleted) time(s)")
                        .font(.subheadline)
                        .foregroundColor(.gray)
                    Text("Calories Burned: \(set.totalCalories) kcal")
                        .font(.subheadline)
                        .foregroundColor(.white)
                }
                .padding()
                .background(Color.gray.opacity(0.2))
                .cornerRadius(12)
            }

            Spacer()
        }
        .padding()
        .background(Color.black.edgesIgnoringSafeArea(.all))
        .navigationTitle("Daily Summary")
    }
}
