import SwiftUI

/// Daily summary view redesigned to emphasize key health metrics
struct DailyDetailView: View {
    @StateObject private var viewModel: DailyDetailViewModel

    init(date: Date) {
        _viewModel = StateObject(wrappedValue: DailyDetailViewModel(date: date))
    }

    var body: some View {
        ZStack {
            Color(white: 0.08).ignoresSafeArea()

            ScrollView {
                VStack(spacing: 32) {
                    heroMetricSection

                    VStack(alignment: .leading, spacing: 16) {
                        Text("Completed Sets")
                            .font(.title3.weight(.bold))
                            .foregroundColor(.white)
                            .padding(.horizontal, 20)

                        LazyVStack(spacing: 12) {
                            ForEach(viewModel.workoutSets) { set in
                                workoutSetCard(set)
                            }
                        }
                        .padding(.horizontal, 20)
                    }
                }
                .padding(.top, 24)
            }
        }
        .navigationTitle(viewModel.date.formatted(.dateTime.weekday().month().day()))
        .navigationBarTitleDisplayMode(.inline)
    }

    private var heroMetricSection: some View {
        VStack(spacing: 8) {
            Image(systemName: "flame.circle.fill")
                .font(.system(size: 48))
                .foregroundStyle(.orange, .orange.opacity(0.2))

            Text("\(viewModel.totalCalories)")
                .font(.system(size: 64, weight: .heavy, design: .rounded))
                .foregroundColor(.white)

            Text("Total Active Kilocalories")
                .font(.headline)
                .foregroundColor(.white.opacity(0.6))
                .textCase(.uppercase)
        }
        .padding(.vertical, 32)
        .frame(maxWidth: .infinity)
        .background(Color.white.opacity(0.03))
        .clipShape(RoundedRectangle(cornerRadius: 24, style: .continuous))
        .padding(.horizontal, 20)
    }

    private func workoutSetCard(_ set: WorkoutSetSummary) -> some View {
        HStack(spacing: 16) {
            VStack(alignment: .leading, spacing: 4) {
                Text(set.name)
                    .font(.headline.weight(.semibold))
                    .foregroundColor(.white)
                Text("\(set.timesCompleted) session\((set.timesCompleted == 1) ? "" : "s")")
                    .font(.subheadline)
                    .foregroundColor(.white.opacity(0.6))
            }
            Spacer()

            Text("\(set.totalCalories) kcal")
                .font(.headline.weight(.semibold))
                .foregroundColor(.yellow)
        }
        .padding(16)
        .background(Color.white.opacity(0.06))
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
    }
}
