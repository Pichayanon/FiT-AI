import Foundation
import SwiftUI

/// Provides formatted data for the post-workout result screen.
@MainActor
final class WorkoutResultViewModel: ObservableObject {
    let summary: SessionSummary

    init(summary: SessionSummary) {
        self.summary = summary
    }

    /// Total session time formatted as "X min Y sec".
    var formattedTotalTime: String {
        WorkoutTextFormatter.minuteSecondString(for: summary.totalTimeSeconds)
    }

    /// Local video file for this session, if the recording is still available on device.
    var recordingURL: URL? {
        SessionVideoStore.recordingURL(for: summary.sessionVideoFileName)
    }
}
