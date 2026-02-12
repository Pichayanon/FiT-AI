import Foundation
import SwiftUI

@MainActor
final class WorkoutResultViewModel: ObservableObject {
    let summary: SessionSummary

    init(summary: SessionSummary) {
        self.summary = summary
    }

    var formattedTotalTime: String {
        let t = summary.totalTimeSeconds
        return "\(t / 60) min \(t % 60) sec"
    }
}
