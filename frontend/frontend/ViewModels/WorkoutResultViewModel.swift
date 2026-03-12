import Foundation
import SwiftUI

@MainActor
final class WorkoutResultViewModel: ObservableObject {
    let summary: SessionSummary

    init(summary: SessionSummary) {
        self.summary = summary
    }

    var formattedTotalTime: String {
        WorkoutTextFormatter.minuteSecondString(for: summary.totalTimeSeconds)
    }
}
