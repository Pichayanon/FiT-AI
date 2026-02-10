import SwiftUI

struct DailyDetailView: View {
    @StateObject private var viewModel: DailyDetailViewModel

    init(date: Date) {
        _viewModel = StateObject(wrappedValue: DailyDetailViewModel(date: date))
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            Text(viewModel.date.formatted(.dateTime.weekday().month().day()))
                .font(.title)
                .foregroundColor(.white)

            Text("Total Burn: \(viewModel.totalCalories) kcal")
                .font(.headline)
                .foregroundColor(.orange)

            Divider().background(Color.gray)

            ForEach(viewModel.workoutSets) { set in
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
